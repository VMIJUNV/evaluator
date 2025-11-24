from dream_evaluator import Evaluator


evaluator = Evaluator.load_from_yaml('test/config.yaml')
evaluator.eval()
