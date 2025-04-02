from Dassl.dassl.utils import Registry, check_availability
from trainers.clip import CLIP
from trainers.promptfl import PromptFL, Baseline
from trainers.pfedmoap import PFEDMOAP

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(CLIP)
TRAINER_REGISTRY.register(PromptFL)
TRAINER_REGISTRY.register(Baseline)
TRAINER_REGISTRY.register(PFEDMOAP)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    # print("avai_trainers",avai_trainers)
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
