# Config loader from Detectron2
from .config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

_C.MODEL.WEIGHTS = ""

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.BASE_LR_PARAM = 0.1
_C.SOLVER.BASE_LR_HYPER = 0.1
_C.SOLVER.NAME_PARAM = 'ADAM'
_C.SOLVER.NAME_HYPER = 'ADAM'
_C.SOLVER.BATCH_SIZE_TRAIN = -1  # Negative batch size means entire dataset.
_C.SOLVER.BATCH_SIZE_VAL = -1
_C.SOLVER.BATCH_SIZE_TEST = -1
_C.SOLVER.UPDATE_PERIOD_HYPER = 100
_C.SOLVER.CHECKPOINT_PERIOD = 5000
_C.SOLVER.MAX_ITER = 5000

# Implicit gradient options
_C.SOLVER.IMPLICIT_GRADIENT_METHOD = 'EXACT'  # 'UNROLL' #'CG'  # NEUMANN, EXACT
# CG
_C.SOLVER.CG = CN()
_C.SOLVER.CG.TOL = 1e-05
_C.SOLVER.CG.MAXITER = 100
# NEUMANN
_C.SOLVER.NEUMANN = CN()
_C.SOLVER.NEUMANN.MAXITER = 100
_C.SOLVER.NEUMANN.ALPHA = 1


_C.TEST = CN()
_C.TEST.EVAL_PERIOD = 5000

# ---------------------------------------------------------------------------- #
# Dataloader options
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
_C.SEED = -1
_C.CUDNN_BENCHMARK = False

# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
