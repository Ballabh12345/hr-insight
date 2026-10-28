"""Model performance - comparison table, curves, calibration study, and global SHAP."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import theme
from app.services import model_service


def _line(x, y, name, color, fill=None, fillcolor=None):
    """Build a Plotly line trace.

    ``fill`` must be one of Plotly's fill enumerations ('tozeroy', 'tonexty', ...);
    ``fillcolor`` carries the translucent colour.
    """
    return go.Scatter(
        x=x, y=y, mode="lines", name=name,
        line={"color": color, "width": 2},
        fill=fill,
        fillcolor=fillcolor,
        hovertemplate=f"{name} %{{y:.3f}}<extra></extra>",
    )


def render() -> None:
    bundle = model_service.load_attrition_metrics()
    if not bundle:
        st.error("No model metrics found. Run `python ml/train_all.py` first.")
        return

    st.markdown(
        theme.app_header(
            "Model performance",
            "How the attrition model was selected, how well it is calibrated, and which "
            "features actually drive the risk.",
        ),
        unsafe_allow_html=True,
    )

    test = bundle["calibrated_test_metrics"]
    st.markdown(
        theme.kpi_row(
            [
                {"label": "ROC-AUC", "value": f"{test['roc_auc']:.3f}"},
                {"label": "PR-AUC", "value": f"{test['pr_auc']:.3f}"},
                {"label": "F1", "value": f"{test['f1']:.3f}"},
                {"label": "Recall", "value": f"{test['recall']:.3f}"},
                {"label": "Precision", "value": f"{test['precision']:.3f}"},
                {"label": "Log-loss", "value": f"{test['log_loss']:.3f}"},
            ]
        ),
        unsafe_allow_html=True,
    )

    bootstrap = bundle.get("bootstrap_ci", {})
    if bootstrap:
        st.markdown(
            f'<div class="small-note">Bootstrap 95% CI (test set, n={bundle.get("dataset", {}).get("test_rows", "?")}): '
            f"ROC-AUC [{bootstrap['roc_auc'][0]:.3f}, {bootstrap['roc_auc'][1]:.3f}] &nbsp;·&nbsp; "
            f"PR-AUC [{bootstrap['pr_auc'][0]:.3f}, {bootstrap['pr_auc'][1]:.3f}].</div>",
            unsafe_allow_html=True,
        )

    conformal = bundle.get("conformal", {})
    if conformal and "q_hat" in conformal:
        st.markdown(
            f'<div class="small-note">Conformal prediction (α={conformal["alpha"]}): '
            f'q̂={conformal["q_hat"]:.3f}, coverage on test={conformal["coverage_test"]:.3f}, '
            f'mean set size={conformal["mean_set_size_test"]:.2f}.</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------------------- model comparison
    st.markdown(theme.section_label("Model comparison"), unsafe_allow_html=True)
    rows = []
    for name, block in bundle["candidates"].items():
        cv = block["cv_roc_auc"]
        rows.append(
            {
                "Model": name,
                "CV ROC-AUC": cv["mean"],
                "Std": cv["std"],
                "Test ROC-AUC": block["test"]["roc_auc"],
                "Test F1": block["test"]["f1"],
                "Test PR-AUC": block["test"]["pr_auc"],
                "Selected": "yes" if name == bundle["selected_model"] else "no",
            }
        )
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            "CV ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
            "Std": st.column_config.NumberColumn(format="%.4f"),
            "Test ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
            "Test F1": st.column_config.NumberColumn(format="%.4f"),
            "Test PR-AUC": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.markdown(
        '<div class="small-note">Selection is by cross-validated ROC-AUC on the training '
        "split only. The test split is evaluated once, at the end.</div>",
        unsafe_allow_html=True,
    )

    # --------------------------------------------- statistical comparison
    paired = bundle.get("paired_tests", {})
    if paired:
        st.markdown(theme.section_label("Statistical comparison vs. selected model"), unsafe_allow_html=True)
        rows = []
        for name, info in paired.items():
            rows.append(
                {
                    "Comparison": f"{bundle['selected_model']} vs {name}",
                    "t-statistic": info.get("t_statistic"),
                    "p-value": info.get("p_value"),
                    "Significant @0.05": "yes" if info.get("significant_at_0.05") else "no",
                }
            )
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            column_config={
                "t-statistic": st.column_config.NumberColumn(format="%.3f"),
                "p-value": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.markdown(
            '<div class="small-note">Nadeau–Bengio corrected paired t-test on the '
            "cross-validation folds. A p-value above 0.05 means the gap is not "
            "statistically distinguishable on this dataset.</div>",
            unsafe_allow_html=True,
        )

    # ------------------------------------------------ feature engineering ablation
    ablation = bundle.get("ablation", {})
    if ablation:
        st.markdown(theme.section_label("Feature engineering ablation"), unsafe_allow_html=True)
        rows = [
            {
                "Variant": "Baseline (raw features)",
                "Features": ablation["baseline"]["n_features"],
                "CV ROC-AUC": ablation["baseline"]["cv_roc_auc"]["mean"],
                "Std": ablation["baseline"]["cv_roc_auc"]["std"],
            },
            {
                "Variant": "With engineered features",
                "Features": ablation["with_engineering"]["n_features"],
                "CV ROC-AUC": ablation["with_engineering"]["cv_roc_auc"]["mean"],
                "Std": ablation["with_engineering"]["cv_roc_auc"]["std"],
            },
        ]
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            column_config={
                "CV ROC-AUC": st.column_config.NumberColumn(format="%.4f"),
                "Std": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.markdown(
            f'<div class="small-note">Delta CV ROC-AUC: '
            f'{ablation["delta_cv_roc_auc"]:+.4f}. Engineered columns: '
            f'{", ".join(ablation.get("engineered_columns", []))}.</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------- curves
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("ROC curve"), unsafe_allow_html=True)
            roc = bundle["roc_curve"]
            fig = go.Figure()
            fig.add_trace(
                _line(
                    roc["fpr"],
                    roc["tpr"],
                    "Model",
                    theme.TOKENS["primary_text"],
                    "tozeroy",
                    "rgba(109, 40, 217, 0.28)",
                )
            )
            fig.add_trace(
                _line([0, 1], [0, 1], "Chance", theme.TOKENS["border_strong"])
            )
            fig.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_xaxes(title_text="False positive rate")
            fig.update_yaxes(title_text="True positive rate")
            theme.chart(fig)
            st.markdown(
                f'<div class="small-note">AUC {roc["auc"]:.4f}</div>',
                unsafe_allow_html=True,
            )

    with right:
        with st.container(border=True):
            st.markdown(theme.section_label("Precision-recall curve"), unsafe_allow_html=True)
            pr = bundle["pr_curve"]
            fig = go.Figure()
            fig.add_trace(
                _line(
                    pr["recall"],
                    pr["precision"],
                    "Model",
                    theme.TOKENS["primary_text"],
                    "tozeroy",
                    "rgba(167, 139, 250, 0.18)",
                )
            )
            fig.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_xaxes(title_text="Recall")
            fig.update_yaxes(title_text="Precision")
            theme.chart(fig)
            st.markdown(
                f'<div class="small-note">Average precision {pr["ap"]:.4f}. '
                "The positive class is only 16% of the workforce, so PR-AUC is the "
                "more informative headline than accuracy.</div>",
                unsafe_allow_html=True,
            )

    # -------------------------------------------------- calibration + matrix
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.markdown(theme.section_label("Calibration"), unsafe_allow_html=True)
            cal = bundle["calibration_curve"]
            fig = go.Figure()
            fig.add_trace(
                _line(
                    cal["prob_pred"], cal["prob_true"], "Model", theme.TOKENS["primary_text"]
                )
            )
            fig.add_trace(_line([0, 1], [0, 1], "Perfect", theme.TOKENS["border_strong"]))
            fig.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8))
            fig.update_xaxes(title_text="Mean predicted probability")
            fig.update_yaxes(title_text="Observed attrition rate")
            theme.chart(fig)

            study = bundle.get("calibration_study", {})
            if study:
                rows = [
                    {
                        "Method": method,
                        "CV log-loss": v["cv_neg_log_loss_mean"],
                        "Std": v["cv_neg_log_loss_std"],
                        "CV Brier": v["cv_neg_brier_mean"],
                    }
                    for method, v in study.items()
                ]
                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "CV log-loss": st.column_config.NumberColumn(format="%.4f"),
                        "Std": st.column_config.NumberColumn(format="%.4f"),
                        "CV Brier": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
                st.markdown(
                    f'<div class="small-note">Selected: <b>{bundle["calibration_method"]}</b>. '
                    "The three methods are statistically indistinguishable on mean log-loss; "
                    "the tie is broken on variance, because isotonic calibration emits "
                    "degenerate 0/1 probabilities on small samples.</div>",
                    unsafe_allow_html=True,
                )

    with right:
        with st.container(border=True):
            st.markdown(theme.section_label("Confusion matrix"), unsafe_allow_html=True)
            cm = bundle["confusion_matrix"]
            fig = go.Figure(
                go.Heatmap(
                    z=cm["normalised"],
                    x=["Predicted: Stay", "Predicted: Leave"],
                    y=["Actual: Stay", "Actual: Leave"],
                    text=[[f"{v}" for v in row] for row in cm["matrix"]],
                    texttemplate="%{text}",
                    textfont={"color": theme.TOKENS["text_primary"]},
                    colorscale=[[i / 4, c] for i, c in enumerate(theme.SEQUENTIAL[:5])],
                    showscale=False,
                    hovertemplate="%{y} / %{x}<br>%{z:.1%} of row<extra></extra>",
                )
            )
            fig.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8))
            theme.chart(fig)
            st.markdown(
                f'<div class="small-note">Decision threshold {bundle["decision_threshold"]:.3f}, '
                "tuned for F1 on out-of-fold training predictions.</div>",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------ survival analysis
    survival = model_service.load_survival_metrics()
    if survival:
        st.markdown(theme.section_label("Survival analysis (time to attrition)"), unsafe_allow_html=True)
        st.markdown(
            theme.kpi_row(
                [
                    {"label": "Train concordance", "value": f"{survival.get('train_concordance', 0):.3f}"},
                    {"label": "Test concordance", "value": f"{survival.get('test_concordance', 0):.3f}"},
                    {"label": "Event rate", "value": f"{survival.get('event_rate', 0):.1%}"},
                ]
            ),
            unsafe_allow_html=True,
        )
        hrs = survival.get("top_hazard_ratios", {})
        if hrs:
            with st.container(border=True):
                st.markdown(theme.section_label("Top hazard ratios (Cox PH)"), unsafe_allow_html=True)
                rows = [{"Feature": k, "Hazard ratio": v} for k, v in list(hrs.items())[:10]]
                st.dataframe(
                    rows,
                    width="stretch",
                    hide_index=True,
                    column_config={"Hazard ratio": st.column_config.NumberColumn(format="%.3f")},
                )

    # ------------------------------------------------------------ SHAP
    st.markdown(
        theme.section_label("What drives attrition risk (global SHAP)"), unsafe_allow_html=True
    )
    importance = model_service.load_shap_global()
    if importance:
        with st.container(border=True):
            top = importance[:15][::-1]
            fig = go.Figure(
                go.Bar(
                    x=[i["importance"] for i in top],
                    y=[i["label"] for i in top],
                    orientation="h",
                    marker={"color": theme.TOKENS["primary_hover"]},
                    hovertemplate="<b>%{y}</b><br>Mean |SHAP| %{x:.4f}<extra></extra>",
                )
            )
            fig.update_layout(height=420, margin=dict(l=8, r=8, t=8, b=8), showlegend=False)
            fig.update_xaxes(title_text="Mean absolute SHAP value (log-odds)")
            theme.chart(fig)
            st.markdown(
                '<div class="small-note">One-hot encoded columns are aggregated back to '
                "their source feature, so each row is one real-world driver.</div>",
                unsafe_allow_html=True,
            )
