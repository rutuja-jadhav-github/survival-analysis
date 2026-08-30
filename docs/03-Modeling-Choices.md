---
layout: default
title: Survival Modelling Algorithms
nav_order: 4
---

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

# Chapter 3. Survival Modelling Algorithms
{: .no_toc }

### Probabilistic vs. Machine Learning Views of Survival Analysis

Survival analysis can be approached from both a **probabilistic/statistical** and a **machine learning** perspective. 

The probabilistic view treats survival time and event occurrence as random variables and seeks to estimate their underlying probabilities or distributions; depending on the method, this may involve strong distributional assumptions (AFT), weaker structural assumptions (Cox), or no specified distribution at all (Kaplan-Meier). 

The machine learning view instead focuses on flexibly learning the relationship between customer features and survival outcomes from data, often using tree-based models to reduce reliance on predefined functional forms.

The sections below are a deeper dive into how both of these angles contribute to this project's pipeline.



## 3.1. Probabilistic/ Statistical Perspective of Survival Analysis

Probabilistic survival modelling is a continuation of the survival theory
developed in Chapter 1.

In Chapter 1, we encountered our first survival estimator: the
**Kaplan–Meier estimator**. KM estimates the probability of surviving beyond
time \(t\), \(S(t)\), by multiplying the conditional probabilities of
surviving each observed event time:

Recall the Kaplan–Meier estimator:

$$
\hat{S}(t)
=
\prod_{t_i \leq t}
\left(
1 - \frac{d_{t_i}}{n_{t_i}}
\right),
$$

where:

- \\(d_{t_i}\\) is the number of churn events at time \\(t_i\\).
- \\(n_{t_i}\\) is the size of the risk set just before \\(t_i\\).

KM provides a population-level survival curve, or separate curves when the
population is divided into strata. However, it does not provide a general
framework for simultaneously conditioning survival on many customer
characteristics.

Probabilistic survival models extend this framework by introducing a
statistical model for how customer characteristics relate to survival.
Three approaches provide a useful progression:

1. **Kaplan–Meier** is **non-parametric**: it does not assume a particular
   mathematical distribution for survival times.

2. **AFT** is generally **parametric**: you specify a distributional form
   for survival/error, such as Weibull or log-normal.

3. **Cox** is **semi-parametric**: it does **not** specify a particular
   distribution for the baseline hazard, but it **does impose the
   proportional hazards assumption** — the effect of a feature is
   multiplicative and stays proportional over time.

The AFT and Cox models therefore allow us to move beyond the population-level
KM curve and incorporate customer characteristics, but they do so under
different statistical assumptions.

### 3.1.1 Accelerated Failure Time (AFT)

The AFT model directly models the relationship between customer
characteristics and the **time until the event occurs**.

The basic idea is that customer features accelerate or decelerate the
underlying survival time. In other words, a feature does not simply increase
or decrease a customer's risk; it changes the **time scale** on which the
event occurs.

A common formulation is:

$$
\log(T)
=
\beta_0
+
\beta_1X_1
+
\cdots
+
\beta_pX_p
+
\epsilon,
$$

where:

- \\(T\\) is the event time.
- \\(X_1,\ldots,X_p\\) are customer features.
- \\(\beta_1,\ldots,\beta_p\\) are model coefficients.
- \\(\epsilon\\) is a random error term.

The coefficients therefore describe how the features affect the expected
event time. For example, a positive coefficient can correspond to a feature
that **extends survival time**, while a negative coefficient can correspond
to a feature that **accelerates the event**.

The AFT model requires an assumed probability distribution for the survival
times or error term, such as a Weibull, log-normal, or log-logistic
distribution. Once this distribution and the model parameters have been
estimated, a customer-specific survival function can be derived.

Conceptually:

$$
X
\quad\longrightarrow\quad
\text{predicted event-time distribution}
\quad\longrightarrow\quad
S(t\mid X)
$$

The key characteristic of AFT is therefore that it models **survival time
directly**, under a specified distributional assumption.

### 3.1.2 Cox Proportional Hazards Model

The Cox model takes a different approach. Rather than modelling survival
time directly, it models the **hazard function** — the instantaneous risk of
experiencing the event at time \\(t\\), given survival up to that point.

The hazard function can be written as:

$$
h(t)
=
\lim_{\Delta t \rightarrow 0}
\frac{
P(t \leq T < t+\Delta t \mid T \geq t)
}{
\Delta t
}.
$$

The Cox model then specifies the hazard for a customer with features \\(X\\)
as:

$$
h(t\mid X)
=
h_0(t)
\exp(\beta_1X_1+\cdots+\beta_pX_p),
$$

where:

- \\(h(t\mid X)\\) is the customer's hazard at time \\(t\\).
- \\(h_0(t)\\) is the **baseline hazard**.
- \\(X\\) represents the customer's features.
- \\(\beta\\) represents the feature coefficients.

The central assumption is the **proportional hazards assumption**:
customer features multiply the baseline hazard by a constant factor over
time.

For example, if two otherwise identical customers have relative risks of
1 and 2, the second customer's hazard is twice that of the first at every
point in time.

Once the baseline hazard has been estimated, the model can be used to obtain
a customer-specific survival function:

$$
S(t\mid X)
=
\exp
\left(
-\int_0^t h(u\mid X)\,du
\right).
$$

Thus, unlike the Kaplan–Meier estimator, the Cox model provides a way to
incorporate multiple customer characteristics into survival modelling.
However, obtaining \\(S(t\mid X)\\) involves additional steps through the
estimated hazard and baseline survival.

### 3.1.3 Why move beyond probabilistic models?

AFT and Cox give us an important capability that KM does not: they allow
survival to depend on customer characteristics.

However, **probabilistic approaches impose a particular statistical structure/ limitations** on the relationship between those characteristics and survival. **AFT requires a distributional assumption, while Cox assumes that feature effects remain proportional** over time.

Customer churn may instead involve complex, non-linear relationships and interactions between features that are difficult to specify in advance.

This motivates the second modelling family considered in this project:
**machine learning approaches**.

The question becomes:

> Can we estimate customer-specific survival, \\(S(t\mid X)\\), while allowing
> the relationship between \\(X\\) and survival to be learned flexibly from
> the data?

This is the motivation for the machine learning models considered next.

----

## 3.2 Machine Learning Perpective of Survival Analysis: Ensemble Methods

The probabilistic models above extend Kaplan–Meier by incorporating customer characteristics, but they do so by imposing a particular statistical structure on the relationship between features and survival.

Machine learning provides an alternative: rather than specifying this relationship in advance, we can allow an ensemble of decision trees to learn complex, non-linear relationships and interactions directly from the data, and THEN compute the survival target like a kaplan meier curve for each customer group.

This section focuses on **pure machine learning ensemble methods on the non-parametric side**: Random Survival Forest (RSF) and XGBSE. The XGBoost survival implementations based on Cox and AFT are not considered further, since they retain the statistical assumptions and formulations discussed in the previous section. They also generate predictions in line with their own respective action spaces (time to churn for AFT, and hazard function for Cox). Generating the complete KM curve after this would require additional steps, that are already embedded in the XGBSE library. 

---

### 3.2.1 Random Survival Forest (RSF)

Random Survival Forests were introduced by Ishwaran et al. (2008) as an extension of random forests to survival outcomes. The key difference from ordinary random forests is that a survival tree does not predict a single scalar outcome at each leaf. Instead, a **Kaplan–Meier survival curve is estimated within each leaf**.


The process is:

1. **Build many decision trees**, with each tree repeatedly splitting
   customers into groups according to their feature values, just as in a
   standard random forest. Each split within a tree is decided based on the resulting child nodes having a substantially different log rank survvial statistic.

2. **Fit a Kaplan–Meier curve in each leaf node**, using the customers that
   reach that leaf.

3. **Pass a new customer through every tree.** The customer's feature values
   determine the path through each tree and therefore the leaf in which the
   customer lands.

4. **Retrieve the leaf-level KM curve** from each tree and average the
   resulting survival curves across the forest.



Random Survival Forest (RSF) removes the strong functional-form assumptions of AFT and Cox.

Instead of specifying:

\\[
\log(T)=\beta^TX+\epsilon
\\]

or:

\\[
h(t\mid X)=h_0(t)\exp(\beta^TX),
\\]

RSF learns relationships directly through an ensemble of survival trees.

For each candidate split, the tree evaluates whether dividing customers into two groups produces substantially different survival experiences, using a survival-specific criterion such as the **log-rank statistic**. Put simply, “If I split these customers at this feature and threshold, do the two resulting groups have noticeably different patterns of survival S(T) over time?”

Once terminal nodes are created, a Kaplan-Meier estimator is calculated within each node.

For a customer \\(X\\), the survival curves from the trees in which that customer lands are then aggregated to obtain \\(
S(t\mid X)\\).

RSF therefore makes no assumption that survival times follow a particular probability distribution or that covariate effects are linear or proportional.

It is consequently a **non-parametric, tree-based survival model**.

Its major advantage for this project is that it directly produces the desired output:

\\[
\boxed{S(t\mid X)}
\\]

while allowing highly non-linear relationships and interactions between customer characteristics.


Conceptually:


$$
\boxed{
X
\quad\longrightarrow\quad
\text{tree splits}
\quad\longrightarrow\quad
\text{leaf-level KM curves}
\quad\longrightarrow\quad
\text{average}
\quad\longrightarrow\quad
S(t\mid X)
}
$$

RSF can therefore be viewed as an augmentation of Kaplan–Meier:

> **KM gives one survival curve for a population or stratum; RSF gives a
> customer-specific survival curve by combining KM curves from leaves
> containing similar customers.**

The similarity between customers is not defined manually. It emerges from the feature-based splits learned by the forest.

Unlike Cox, RSF does not require the proportional hazards assumption. Because the trees can split on features repeatedly and at different thresholds, the model can also capture **non-linear feature effects and interactions** without specifying them in advance.

This makes RSF particularly useful for the present problem: the objective is to estimate \\(S(t\mid X)\\)


directly while allowing the relationship between customer characteristics and survival to be learned flexibly from the data.

---


### 3.2.2 XGBSE

Following the Random Survival Forest (RSF), the second tree-based survival model considered for this project was XGBSE (XGBoost Survival Embeddings).

### Why XGBSE instead of XGBoost Cox or XGBoost AFT?

XGBoost provides two native survival objectives:

- **Cox proportional hazards (`survival:cox`)**
- **Accelerated Failure Time (`survival:aft`)**

Both approaches use gradient-boosted decision trees and can model complex, non-linear relationships between customer characteristics and survival outcomes. However, their outputs are fundamentally different from the output required in this project.

#### XGBoost Cox

The Cox objective estimates a customer's **relative risk score**:

\\[
h(t|X)=h_0(t)\exp(f(X))
\\]

where \\(f(X)\\) is learned using boosted trees.

The model therefore answers:

> How much greater or lower is this customer's risk of churn relative to others?

The output is a risk score rather than a full survival curve.

---

#### XGBoost AFT

The AFT objective models survival time directly:

\\[
\log(T)=f(X)+\epsilon
\\]

where \\(f(X)\\) is again learned using boosted trees.

The model therefore answers:

> How long is this customer expected to survive before churn?

The output is a predicted survival-time distribution rather than a full customer-specific survival curve.

---

### Why neither Cox nor AFT alone was sufficient

The objective of this project is not merely to rank customers by risk or predict a single expected churn time.

Instead, the goal is to estimate the complete survival function:

\\[
S(t|X)
\\]

which provides the probability that a customer remains active beyond any future time \\(t\\).

While XGBoost Cox and XGBoost AFT both learn useful survival representations, additional steps are required to transform their outputs into a full survival curve.

XGBSE was therefore selected because it directly produces customer-level survival curves while still benefiting from the predictive power of gradient-boosted trees.

---

### How XGBSE Works

XGBSE can be viewed as a two-stage model:

\\[
X
\rightarrow
\text{XGBoost Survival Model}
\rightarrow
\text{Customer Embedding}
\rightarrow
\text{Kaplan-Meier Estimation}
\rightarrow
S(t|X)
\\]

For this project, the **XGBSEKaplanNeighbors** implementation was used with the **AFT survival objective**.

---

### Stage 1: XGBoost AFT Tree Construction

As with any decision-tree model, the first step is determining how observations should be split.

Different tree models use different criteria:

| Model | Split Criterion |
|---------|----------------|
| Regression Tree | Mean Squared Error (MSE) |
| Classification Tree | Gini Impurity / Entropy |
| Random Survival Forest | Log-rank statistic |
| XGBSE (this project) | AFT negative log-likelihood |

The survival outcome consists of:

\\[
(T,\delta)
\\]

where:

- \\(T\\) = observed customer tenure
- \\(\delta = 1\\) if churn occurred
- \\(\delta = 0\\) if the customer is censored

Using these survival outcomes, XGBoost sequentially builds trees that maximize improvement in the AFT objective.

Conceptually:
$$
\boxed{\displaystyle
(T,\delta)
\rightarrow
\text{AFT Loss}
\rightarrow
\text{Best Feature Split}
\rightarrow
\text{Boosted Trees}
}
$$

Unlike RSF, which directly uses a survival-specific log-rank statistic to choose splits, XGBSE uses the gradient boosting framework to identify feature thresholds that improve the AFT likelihood.

The resulting trees therefore group together customers with similar survival behaviour according to the AFT objective.

---

### Stage 2: Kaplan Neighbors

Once the boosted trees have been trained, XGBSE no longer focuses on the AFT predictions themselves.

Instead, it examines how customers move through the tree ensemble.

Consider five trees:

| Customer | Tree 1 | Tree 2 | Tree 3 | Tree 4 | Tree 5 |
|----------|--------|--------|--------|--------|--------|
| A | 2 | 4 | 1 | 7 | 3 |
| B | 2 | 4 | 1 | 7 | 3 |
| C | 2 | 4 | 5 | 7 | 3 |
| D | 8 | 6 | 9 | 1 | 4 |

Customers A and B land in the same terminal node in every tree.

Customers A and C land together in four out of five trees.

Customer D follows a completely different path through the ensemble.

XGBSE uses these leaf assignments to create a tree-based embedding of each customer.

Customers that repeatedly co-occur in the same terminal nodes are considered similar.

Similarity is measured using Hamming distance across the leaf assignments.

For a new customer:

```python
n_neighbors = 30
```

the model identifies the 30 most similar training customers in this learned embedding space.

The observed survival outcomes of those neighbours are then used to construct a Kaplan-Meier estimator:

\\[
30\ \text{Nearest Neighbours}
\rightarrow
(T,\delta)
\rightarrow
\text{Kaplan-Meier}
\\]

producing the final customer-specific survival function:

\\[
S(t|X)
\\]

---

## 3.3 Summary: From Kaplan-Meier to XGBSE

The survival models considered in this project range from a purely non-parametric estimator to increasingly flexible tree-based approaches. Although all five models ultimately address the same underlying question — *what is the probability that a customer remains active beyond time \(t\)?* — they differ substantially in how they represent the relationship between customer characteristics, time-to-churn, and the resulting survival curve.

### 3.3.1 Parametric vs. Non-Parametric: An Important Distinction

**Parametric** - A model is **parametric** when it makes a specific assumption about the mathematical form or probability distribution underlying the outcome.

In this project, for instance, the AFT objective is parametric:

$$
\log(T)=F(X)+\epsilon
$$

and, because we selected the normal distribution:

$$
\epsilon\sim N(0,\sigma^2)
$$

So the AFT objective makes a **parametric distributional assumption about log survival time**.In other words, it assumes that the residual variation around the predicted log-survival time follows a particular distribution.


**Non-parametric** - A non-parametric approach does not require survival times to follow a particular predefined probability distribution. For example, the Kaplan-Meier estimator does not assume that survival times are normally distributed, Weibull distributed, etc. It estimates the survival curve directly from the observed data simply as \\(S(t)\\)


**Non-Parametric Caveat for XGBSE** - In XGBSE, the **AFT objective used to build the trees is parametric in its survival-time assumption**. However **final Kaplan-Meier estimation** in the target space does **not** require the final output, i.e, the survival curves of the neighbour groups to follow any distribution, and is therefore non parametric.

> **The AFT objective is parametric, but the final XGBSE survival estimation is not simply a parametric AFT prediction.**

----

### 3.3.2. Survival Models and Algorithms Summarised

A useful way to distinguish the models is therefore by asking two questions:

1. **What does the model learn from the data?**
2. **What assumptions does it make about the relationship between covariates and survival?**

The models considered are:

1. Kaplan-Meier (in scope - target space)
2. Parametric AFT (out of scope)
3. Cox Proportional Hazards (out of scope)
4. Random Survival Forest (in scope - action/hypothesis space)
5. XGBSE Kaplan Neighbors with an AFT objective  (in scope - action/hypothesis space)

#### MODEL COMPARISON

| Model | Main purpose | How customer characteristics enter the model | Main output | Parametric structure |
|---|---|---|---|---|
| **Kaplan-Meier** | Estimate overall survival | No covariates required | \\(S(t)\\) | **Non-parametric** |
| **AFT** | Model time-to-event | Explicit statistical relationship between \\(X\\) and \\(\log(T)\\) | Survival-time distribution / \\(S(t\mid X)\\) | **Parametric** |
| **Cox PH** | Model relative event risk | Proportional-hazards relationship between \\(X\\) and risk | Relative risk / hazard ratio | **Semi-parametric** |
| **Random Survival Forest** | Learn complex survival patterns | Survival-specific tree splits using log-rank separation | \\(S(t\mid X)\\) | **Non-parametric** |
| **XGBSE Kaplan Neighbors** | Learn complex customer survival groups | Boosted trees trained using an AFT objective, followed by neighbour-based KM estimation | \\(S(t\mid X)\\) | **Hybrid / semi-parametric** |

---
The figure below summarises the key topics I studied for this project, almost like a mind-map. As clarified in this chapter, the survival models explored belong to this path: 

> **Customer Features -> Tree-Based -> RSF and XGBSE -> S(t\|X)**

ie, we basically help address the 'no feature, generic curve' limitation of an ordinary kaplan meier curve, by augmenting it with features using Tree-based Ensemble ML models. 



                                  SURVIVAL ANALYSIS
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
              NO CUSTOMER FEATURES                     CUSTOMER FEATURES
                    │                                         │
                    ▼                                         ▼
              ┌───────────┐                         ┌──────────────────┐
              │ Kaplan-   │                         │ Statistical or   │
              │  Meier    │                         │  Tree-based?     │
              └───────────┘                         └────────┬─────────┘
              Non-parametric                                  │
              Overall S(t)                          ┌─────────┴─────────┐
                                                    │                   |
                                             STATISTICAL               TREE-BASED
                                                   │                       │
                                           ┌───────┴───────┐          ┌────┴────┐
                                           │               │          │         │
                                          AFT             Cox        RSF      XGBSE
                                           │               │          │         │
                                      Parametric      Semi-param.     │         Hybrid
                                      Time model       Risk model     │        AFT objective+embedding
                                           │               │          │        + KM
                                           ▼               ▼          ▼         S(t|X), ie, feature-augmented KM curve
                                      Extra step                      S(t|X)
                                      for S(t|X)        Risk score    ie, feature-augmented     
                                                           │          KM curve
                                                           ▼          
                                                  Extra step required
                                                  for full S(t|X)