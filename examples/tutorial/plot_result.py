import rlpy.Tools.results as rt

paths = {"Tabular": "./Results/Tutorial/gridworld-qlearning"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps", "return")
rt.save_figure(fig, "./Results/Tutorial/plot.pdf")
