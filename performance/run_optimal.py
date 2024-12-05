"""Performance of running the optimal algorithm."""
import cProfile

from incomplete_cooperative.__main__ import main

args = ['--number-of-players=4', '--game-generator=factory', '--game-class=superadditive_cached', '--parallel-environments=5', '--run-steps-limit=10', '--policy-activation-fn=tanh', 'best_states', '--sampling-repetitions', '2']


cProfile.run(f"main(args={args})", "fast.prof")
