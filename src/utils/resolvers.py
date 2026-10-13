import os

import hydra
import omegaconf
from omegaconf import OmegaConf



def register_resolvers():
    OmegaConf.register_new_resolver("hydra_module", lambda: hydra)
    OmegaConf.register_new_resolver("omegaconf_module", lambda: omegaconf)
    OmegaConf.register_new_resolver("local_username", os.getlogin)