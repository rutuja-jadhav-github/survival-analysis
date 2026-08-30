---
layout: default
title: About the Project
nav_order: 1
---

# Augmenting Kaplan–Meier Survival Analysis with Ensemble Methods for Customer Churn Prediction.
{: .no_toc }

<div class="author-block">
  <div class="author-name">Rutuja Jadhav</div>
  <a href="https://rutujajadhavmail2.wixsite.com/rutujajadhav/copy-of-projects" target="_blank">
    <i class="fa-solid fa-globe"></i> Website
  </a>
  <a href="https://www.linkedin.com/in/rutuja-jadhav-b96b2aa7/" target="_blank">
    <i class="fa-brands fa-linkedin"></i> LinkedIn
  </a>
  <a href="https://github.com/rutuja-jadhav-github/Beyond-Markowitz-Optimisation-RF-GBM" target="_blank">
    <i class="fa-brands fa-github"></i> GitHub
  </a>
</div>


## About the Project

This project was developed as part of my capstone for DS-GA 1003 Machine Learning during my M.S. in Data Science at New York University's Center for Data Science, Courant Institute. It is based on applying survival analysis to the telecommunications market using the Telco Customer Churn dataset, with the goal of predicting customer churn over time.

The original version is more of an academic piece confined to a Jupyter notebook and about 8 pages of a journal-style paper. The version presented here is a much deeper dive into survival theory and modeling approaches, followed by preliminary deployment using Streamlit + REST API + basic docker/ conternisation pipeline to build an app that predicts survival probabilities for customers in the telecommunications market using the aforementioned data from Telco.



## Introduction to Survival Analysis

Survival analysis originated in clinical medicine, where researchers needed to answer a deceptively simple question: how long until a patient dies? The challenge was not the question but the data — patients dropped out of studies, studies ended before all patients died, and some patients were still alive at the time of analysis. Simply averaging the observed times would throw away real information. Simply labelling surviving patients as “deceased” would fabricate outcomes. 

The field developed principled methods to handle this uncertainty, most notably the Kaplan–Meier estimator [Kaplan and Meier, 1958], and the Cox proportional hazards model [Cox, 1972]. The same mathematical problem appears in finance (time to loan default [Saavedra et al., 2024]), insurance (time to claim), and marketing (time to churn). A customer who is still active today is the direct analogue of a patient still alive at study end: their story is unfinished, their outcome unknown, but the time they have already survived is real information that belongs in the analysis.


## Research Questions/ Contributions

In this project, we shall approach churn prediction as a survival analysis problem, beginning with the Kaplan–Meier (KM) estimator as a censoring-aware baseline, and augmenting it with two ensemble methods: Random Survival Forest (RSF) and XGBoost Survival Embeddings (XGBSE). 

This project makes the following contributions:

1. Demonstrates empirically that binary classification corrupts churn labels of customers in the Telco dataset, systematically underestimating churn risk (ie, overestimating survival) in the early tenure band.

2. Shows how the Kaplan–Meier estimator handles censoring honestly but produces one population-level curve with no feature conditioning.

3. Augments KM with two ensemble methods — RSF [Ishwaran et al., 2008] and XGBoost Survival — that produce per-customer/per-group survival curves conditioned on features, evaluated with concordance index and integrated Brier score.

4. The project further manifests into a more usable format with a basic Streamlit UI using two modes:
- **Mode A — Batch analysis**: upload a CSV of customers, set a churn probability threshold based on your business requirements and a time window, get back the customers to target for a retention campaign.
- **Mode B — Single customer**: fill in one customer's details, get a Low/Medium/High risk label and a survival curve plotted against example risk profiles.

5. This is followed by preliminary ML Ops/ deployment where in I used a REST API to faciliate availabiilty of the applicaiton to all other kinds of frontends apart from stream lit - example another team's website, internal app, mobile app, etc, and loaded it onto a docker and container as a single unit enabling to operate as a complete unit of its own. 


## Dataset and Project Pipeline

This analysis uses the IBM Telco Customer Churn dataset [IBM], comprising 7,032 customers after removing 11 records with
missing TotalCharges (all with tenure = 0, carrying no survival information). The dataset has a 26.6% churn proportion with
customer tenures ranging from 1 to 72 months. Customers with Churn= No within a given window are treated as right-censored observations.

```
EDA Insights
     │
     ▼
KM Estimator vs Binary Classification
     │
     ▼
Survival Model Algorithms
(KM, AFT, Cox, RSF, XGBSE)
     │
     ▼
ML Build & Evaluation
(RSF & XGBSE with KM Benchmark)
     │
     ▼
Defining Risk Tiers
     │
     ▼
ML Deployment
     │
     ▼
Limitations & Future Scope

```

## Key Findings
On the IBM Telco Customer Churn dataset (7,032 customers), RSF achieves a concordance index of 0.8480 and an integrated Brier score of 0.0932, while XGBSE achieves a concordance index of 0.8471 and an integrated Brier score of 0923. Both substantially outperform the KM baseline while producing per-customer survival curves that enable time-stamped, actionable retention decisions.

