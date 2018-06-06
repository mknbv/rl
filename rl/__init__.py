import logging

logger = logging.getLogger("rl")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
logger.addHandler(stream_handler)

import rl.algorithms
import rl.policies
import rl.data
import rl.training

from rl.utils import env_batch
from rl.utils import env_wrappers
from rl.utils import launch_utils
from rl.utils import tf_utils
