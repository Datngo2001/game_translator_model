import tensorflow as tf
from datasets import load_dataset

dataset = load_dataset("yhavinga/ccmatrix", "en-nl", streaming=True, trust_remote_code=True)
