"""
Module providing visualisations
"""
import shap
import plotly
import kaleido
import matplotlib.pyplot as plt
import seaborn as sns
from interpret import show

from preparation import *


shap.initjs()
sns.set(font_scale=1.7)


def show_confusion_matrix(conf1, conf2, categories, save_name):
    """
    Function to plot compare confusion matrices of the two models
    :param conf1: the first confusion matrix
    :param conf2: the second confusion matrix
    :param categories: names of the positive and negative categories
    :param save_name:save the plot with this name
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))
    xlabels = ["true " + cat for cat in categories]
    ylabels = ["predicted " + cat for cat in categories]
    sns.heatmap(conf1, ax=axs[0], cmap="flare", annot=True, cbar=False, xticklabels=xlabels,
                yticklabels=ylabels, fmt="g")
    sns.heatmap(conf2, ax=axs[1], cmap="flare", annot=True, cbar=True, xticklabels=xlabels,
                yticklabels=ylabels, fmt="g")
    plt.savefig(save_name)
    plt.show()


def explain_shap(model, Xtest):
    """
    Use SHAP values to produce summary plot and explain one prediction on a dataset
    :param model: model to visualise
    :param Xtest: test data
    """
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(Xtest)
    shap.summary_plot(shap_vals, Xtest, plot_type="bar", plot_size=(12, 10), show=False)
    plt.savefig("../figs/shap_summary.pdf")
    plt.show()
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.savefig("../figs/shap_waterfall.pdf", bbox_inches='tight')
    plt.show()


def interpret_ebm(model, name, Xtest, ytest):
    """
    Visualise EBM with a summary plot and then one feature function
    :param model: model to visualise
    :param name: prefix for the names of the saved plots
    :param Xtest: test data
    :param ytest: test labels
    """
    exp_global = model.explain_global()
    # interactive plot in the notebook
    show(exp_global)
    # save the first feature plot using kaleido
    fig = exp_global.visualize(0)
    fig.write_image(f"../figs/{name}_ebm_col.pdf")
    exp_local = model.explain_local(Xtest, ytest)
    # interactive plot in the notebook with explanations for each prediction
    show(exp_local)
    # save the one local explanation plot using kaleido
    fig = exp_local.visualize(0)
    fig.write_image(f"../figs/{name}_ebm_pred.pdf")
    # recreate the feature importance plot and save it
    feature_names = model.term_names_
    feature_importance = model.term_importances()
    df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
    df.sort_values(by=["importance"], ascending=False, inplace=True)
    fig = plt.figure(figsize=(16, 10))
    sns.barplot(df, x="importance", y="feature", orient="y")
    plt.savefig(f"../figs/{name}_ebm_fi.pdf", bbox_inches='tight')
    plt.close(fig)