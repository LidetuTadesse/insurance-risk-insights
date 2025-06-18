import shap

def explain_with_shap(model, X_sample):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values, X_sample, plot_type="bar")