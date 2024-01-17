import optuna
import matplotlib.pyplot as plt
import optuna.visualization as vis
import plotly.io as pio

study_name = 'SorscherRNN_study'
study_storage = 'sqlite:///tuning_alpha_beta.db'

study = optuna.load_study(study_name=study_name, storage=study_storage)

fig = vis.plot_contour(study, params=['alpha', 'beta'])

fig['data'][0]['colorscale'] = 'Viridis'

# To save the figure, you can use the following command:
# The file format will be inferred from the file extension. Supported formats are jpeg, png, svg, and pdf.
pio.write_image(fig, './fig/contour_plot.png')

fig = vis.plot_optimization_history(study)
fig.write_image('./fig/optimization_history.png')

fig = vis.plot_param_importances(study)
fig.write_image('./fig/param_importances.png')

#correlation
fig = vis.plot_parallel_coordinate(study)
fig.write_image('./fig/parallel_coordinate.png')



#plot the best trial attr (loss_history)
best_trial = study.best_trial
loss_history = best_trial.user_attrs["loss_history"]
plt.plot(loss_history[::100])
plt.ylabel("Loss")
plt.savefig("./fig/loss_history_best_trial.png")