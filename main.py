from src.train.trainer import Trainer

t = Trainer((0,-1),(-1,None),"proxy")
t.fit()

from src.eval.validator import run_evaluate

run_evaluate(t)