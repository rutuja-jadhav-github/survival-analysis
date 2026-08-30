---
layout: default
title: Limitations and Future Scope
nav_order: 8
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

# Chapter 7. Limitations and Future Scope
{: .no_toc }

Despite its humble classical roots (no fancy deep learning), this was hands down one of my most favourite projects at uni. It helped me explore multiple avenues in one place - the probabilistic angle (beliefs about how the survival universe works), the ML view (how the predictive models work) and even the tip of the deployment/ system design iceberg. (Thankfully, I did not end up disliking the system design bits as much as I thought I would, mainly because it helped me catch an important bug in my model's logic!).

While there are always several improvements that can be made, following are the major learning curves and opportunities worth a shout:

**1. Class imbalance (26% churn rate).** The dataset's ~26% churn rate is a mild imbalance, not a severe one (severe would be closer to single digits, like fraud detection). It's unlikely to meaningfully distort ranking metrics like C-index, which is relatively robust to imbalance. The more realistic risk is at the leaf level: nodes with few observed churn events produce noisier per-leaf KM curves, since there's less direct evidence to estimate the curve from. This is worth monitoring rather than treating as a solved non-issue — a useful future check would be stratifying evaluation by tenure band to confirm the model isn't systematically less confident on shorter-tenure customers, where churn events are sparser.

**2. Logic bug in model.py - churn and survival probabilities not summing to 1.** During deployment, the app briefly reported inconsistent numbers — the displayed churn probability and the plotted survival curve didn't agree at the same point in time. The cause: the survival curve's data only starts at month 1 (no customer in the dataset has zero tenure - we also dropped 11 rows with zero tenure as their Tot charges was NaN = monthly charges *0). 

Thus, evaluating the curve at exactly t=0 for a brand-new customer probably returned the month-1 value instead of the mathematically correct certainty of 1.0 (nobody can churn before any time has passed). This meant every brand-new-customer prediction was potentially overstating churn risk. The fix explicitly sets survival probability to 1.0 for any time before the curve's earliest known data point, rather than relying on the underlying library's default extrapolation.Worth calling out as a reminder that curve-based calculations deserve an explicit sum-to-one sanity check, not just visual inspection of a plot.

**3. Conditioning on current tenure.** The deployed app currently treats every customer as brand new (tenure = 0) when computing churn probability, even though the underlying logic supports conditioning on a customer's actual elapsed tenure for a more accurate probability (`P(churn within M months | already survived t0 months)`). This conditional approach is implemented and demonstrated in a standalone notebook (`demo_conditional_vs_unconditional.ipynb`) but isn't yet wired into the app's logic or UI — a natural next step for anyone extending this project. See the notebook for the full derivation and a worked comparison.

