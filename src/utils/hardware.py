"""
Module de gestion et d'abstraction du matériel d'accélération (CUDA, MPS, Intel XPU, CPU).

Ce module comble l'absence de support natif pour le backend Intel XPU dans PyTorch Lightning 2.x
et applique les correctifs bas niveau requis pour le runtime Intel oneAPI Level-Zero :
1. Enregistrement de 'xpu' parmi les types de périphériques à transferts bloquants (synchrones)
   afin de prévenir les corruptions de mémoire lors du déplacement des batches.
2. Définition de l'accélérateur `XPUAccelerator` conforme aux interfaces Lightning.
3. Définition du plugin de précision `XPUPrecision` désactivant les kernels multi-tenseurs
   (`foreach=False`) lors du clipping de gradient, instables sur le pilote Level-Zero Windows.
4. Définition de la stratégie `SingleXPUStrategy` avec synchronisation explicite des files de
   commandes SYCL (`torch.xpu.synchronize()`).
5. Détection dynamique du matériel pour assurer une compatibilité totale et transparente
   sur tous les environnements (NVIDIA CUDA, Apple Silicon MPS, Intel Arc XPU, CPU).
"""

import logging
from typing import Any, List, Sequence, Union
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device
import lightning.fabric.utilities.apply_func as apply_func_module
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.plugins.precision import Precision
from lightning.pytorch.strategies import SingleDeviceStrategy

# ---------------------------------------------------------------------------
# Patch du mécanisme de transfert asynchrone de PyTorch Lightning Fabric
# ---------------------------------------------------------------------------
# Par défaut, Lightning applique `tensor.to(device, non_blocking=True)` pour tout périphérique
# qui n'est pas dans `_BLOCKING_DEVICE_TYPES` (qui ne contient nativement que 'cpu' et 'mps').
# Sur le runtime Intel Level-Zero sous Windows, un transfert asynchrone sans synchronisation de
# flux engendre une condition de concurrence où les mémoires partagées peuvent être écrasées
# avant consommation (ex. corruption des étiquettes cibles par les coordonnées 3D).
# L'ajout de 'xpu' garantit l'intégrité stricte des transferts hôte-GPU.
if hasattr(apply_func_module, "_BLOCKING_DEVICE_TYPES") and "xpu" not in apply_func_module._BLOCKING_DEVICE_TYPES:
    apply_func_module._BLOCKING_DEVICE_TYPES = tuple(apply_func_module._BLOCKING_DEVICE_TYPES) + ("xpu",)

log = logging.getLogger(__name__)


class XPUAccelerator(Accelerator):
    """
    Accélérateur PyTorch Lightning pour les GPU Intel (Intel Arc, Core Ultra iGPU, Data Center Max).

    Fait le pont entre l'abstraction matérielle de Lightning et l'API `torch.xpu` introduite
    officiellement dans PyTorch 2.4+.
    """

    def setup_device(self, device: torch.device) -> None:
        """Configure le périphérique actif pour le thread courant."""
        torch.xpu.set_device(device)

    def teardown(self) -> None:
        """Nettoyage en fin d'entraînement (aucun état persistant requis pour XPU)."""
        pass

    @staticmethod
    def parse_devices(devices: Any) -> List[int]:
        """Convertit l'argument de configuration des périphériques en liste d'indices."""
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, (list, tuple)):
            return list(devices)
        return [0]

    @staticmethod
    def get_parallel_devices(devices: Sequence[int]) -> List[torch.device]:
        """Génère la liste des objets `torch.device` correspondants."""
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        """Retourne le nombre total de périphériques XPU Intel disponibles."""
        return torch.xpu.device_count() if hasattr(torch, "xpu") else 0

    @staticmethod
    def is_available() -> bool:
        """Vérifie la disponibilité opérationnelle du backend XPU."""
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def name() -> str:
        """Identifiant unique de l'accélérateur."""
        return "xpu"


class XPUPrecision(Precision):
    """
    Plugin de précision adapté à l'architecture Intel XPU et Level-Zero.

    Sous PyTorch 2.x, `torch.nn.utils.clip_grad_norm_` utilise par défaut `foreach=True`
    dès qu'un accélérateur non-CPU est détecté, invoquant des kernels vectoriels
    (`torch._foreach_norm` et `torch._foreach_mul_`).
    Sur les pilotes Intel Level-Zero Windows actuels, ces kernels provoquent des corruptions
    internes conduisant à l'erreur critique UR_RESULT_ERROR_DEVICE_LOST (code 20).
    Cette classe force l'évaluation séquentielle unitaire (`foreach=False`), stable et performante.
    """

    def clip_grad_by_norm(self, optimizer: Any, clip_val: Union[float, int]) -> None:
        """Tronque les gradients en norme en désactivant le backend multi-tenseurs."""
        parameters = self.main_params(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, clip_val, foreach=False)


class SingleXPUStrategy(SingleDeviceStrategy):
    """
    Stratégie d'exécution mono-périphérique dédiée aux GPU Intel.

    Prend en charge :
    - L'injection automatique de `XPUAccelerator` et de `XPUPrecision`.
    - Le transfert synchrone des batches vers la mémoire GPU.
    - La vidange et la synchronisation explicite des files de commandes SYCL
      (`torch.xpu.synchronize()`) entre les passes d'entraînement et les étapes
      d'optimisation afin d'éviter l'épuisement des descripteurs de commandes Level-Zero
      (erreur UR_RESULT_ERROR_OUT_OF_RESOURCES, code 40).
    """

    def __init__(self, device: Union[int, str, torch.device] = "xpu:0", **kwargs: Any) -> None:
        if isinstance(device, int):
            device = torch.device(f"xpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        kwargs.setdefault("precision_plugin", XPUPrecision())
        super().__init__(device=device, accelerator=XPUAccelerator(), **kwargs)

    def batch_to_device(
        self, batch: Any, device: Union[torch.device, None] = None, dataloader_idx: int = 0
    ) -> Any:
        """Transfère de manière sécurisée et synchrone une structure de données vers le GPU XPU."""
        target_device = device or self.root_device
        return move_data_to_device(batch, target_device)

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """Exécute l'étape d'entraînement puis synchronise la file SYCL."""
        loss = super().training_step(*args, **kwargs)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        return loss

    def optimizer_step(self, optimizer: Any, closure: Any, model: Any = None, **kwargs: Any) -> Any:
        """
        Exécute la mise à jour des poids par l'optimiseur avec synchronisation amont et aval
        pour garantir la cohérence des buffers de gradients et d'états d'optimisation.
        """
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        result = super().optimizer_step(optimizer, closure, model=model, **kwargs)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        return result


def get_hardware_strategy(cfg_trainer: Any = None) -> Union[SingleXPUStrategy, None]:
    """
    Détecte automatiquement l'infrastructure matérielle disponible et configure la stratégie adaptée.

    Hiérarchie de sélection :
    1. Si une stratégie ou un accélérateur explicite est déjà configuré dans Hydra (ex. cluster SLURM),
       la détection automatique s'efface pour respecter la configuration utilisateur.
    2. NVIDIA CUDA : Lightning prend en charge CUDA nativement, aucune stratégie personnalisée requise.
    3. Apple Silicon (MPS) : Lightning prend en charge MPS nativement, aucune stratégie personnalisée requise.
    4. Intel XPU (Arc / Core Ultra) : Injection de `SingleXPUStrategy` pour combler le manque natif de Lightning.
    5. Défaut : Retourne None pour laisser Lightning exécuter le modèle sur CPU.

    :param cfg_trainer: Section de configuration Hydra du Trainer (optionnelle).
    :return: Une instance de SingleXPUStrategy si un GPU Intel est détecté, None sinon.
    """
    if cfg_trainer and cfg_trainer.get("strategy") is not None:
        return None

    if cfg_trainer and cfg_trainer.get("accelerator") not in (None, "auto"):
        return None

    # 1. Environnement NVIDIA CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"[Hardware] Accélérateur NVIDIA CUDA détecté ({gpu_name}). Utilisation du backend natif.")
        return None

    # 2. Environnement Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.info("[Hardware] Accélérateur Apple Silicon MPS détecté. Utilisation du backend natif.")
        return None

    # 3. Environnement Intel XPU (Intel Arc / Core Ultra / Data Center Max)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device_name = torch.xpu.get_device_name(0)
        log.info(f"[Hardware] Accélérateur Intel XPU détecté : {device_name}. Activation de SingleXPUStrategy.")
        return SingleXPUStrategy()

    # 4. Fallback CPU
    log.info("[Hardware] Aucun accélérateur GPU détecté. Exécution sur CPU.")
    return None
