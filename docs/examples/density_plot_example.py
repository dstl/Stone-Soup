from datetime import datetime, timedelta
from matplotlib import pyplot as plt


from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.plotter import Plotter

start_time = datetime.now()


def generate_ground_truth_path(initial_state, num_steps=20, q=0.01):

    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q), ConstantVelocity(q)])

    ground_truth = GroundTruthPath([GroundTruthState(initial_state, timestamp=start_time)])

    for k in range(0, num_steps):
        ground_truth.append(GroundTruthState(
            transition_model.function(ground_truth[k], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k+1)))

    return ground_truth


n_time_steps = 20
truth = generate_ground_truth_path(initial_state=[0, 0, 0, 1], num_steps=n_time_steps)

Plotter().plot_ground_truths(truth, [0, 2])
plt.show()

truths = [generate_ground_truth_path(initial_state=[0, 0, 0, 1], num_steps=n_time_steps, q=0.1) for _ in range(100)]

# This is quite messy
Plotter().plot_ground_truths(set(truths), [0, 2])
plt.show()

# This is clearer
Plotter().plot_density(truths, index=None)
plt.show()

# Only show the last state
Plotter().plot_density(truths, index=-1)
plt.show()

plotter = Plotter()
for i in range(1, n_time_steps):
    plotter.plot_density(truths, index=i)
    plt.show(block=False)
    plt.pause(0.1)

plt.show()
