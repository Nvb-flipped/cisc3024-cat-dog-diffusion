# Check 3 – Hand calculation: add_noise and reconstruct_clean

Date: 2026-11-22
Time: 20:28–20:37 (GMT+8)
Author: Sou Chak Tong
Method: worked through by hand, with a calculator for the square roots and decimals.

## Purpose

Verify by hand that the diffusion helper functions in `submission/diffusion.py`
perform the standard DDPM arithmetic:

\[
x_t = \sqrt{\bar\alpha}\,x_0 + \sqrt{1-\bar\alpha}\,\epsilon
\]

and

\[
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar\alpha}\,\epsilon}{\sqrt{\bar\alpha}}
\]

## Inputs

- Clean value: \(x_0 = 0.4\)
- Cumulative alpha: \(\bar\alpha = 0.81\)
- Noise: \(\epsilon = -0.2\)

## Step 1: Compute the noisy value

First compute the two coefficients:

\[
\sqrt{\bar\alpha} = \sqrt{0.81} = 0.9
\]

\[
\sqrt{1-\bar\alpha} = \sqrt{1-0.81} = \sqrt{0.19}
\approx 0.4358898943540674
\]

Now compute \(x_t\):

\[
x_t = 0.9 \times 0.4
    + 0.4358898943540674 \times (-0.2)
\]

\[
x_t = 0.36 - 0.08717797887081348
\]

\[
x_t \approx 0.2728220211291865
\]

So the noisy value is:

\[
\boxed{x_t \approx 0.2728220211}
\]

## Step 2: Recover the clean value from the noisy value

Use:

\[
\hat{x}_0 =
\frac{x_t - \sqrt{1-\bar\alpha}\,\epsilon}{\sqrt{\bar\alpha}}
\]

Substitute the values:

\[
\hat{x}_0 =
\frac{
0.2728220211291865
-
0.4358898943540674 \times (-0.2)
}{0.9}
\]

\[
\hat{x}_0 =
\frac{
0.2728220211291865
+
0.08717797887081348
}{0.9}
\]

\[
\hat{x}_0 =
\frac{0.36}{0.9}
= 0.4
\]

So the recovered clean value is:

\[
\boxed{\hat{x}_0 = 0.4}
\]

This matches the original clean value \(x_0 = 0.4\).

## Step 3: Compare with `submission/diffusion.py`

The two functions in the submitted code are:

```python
def add_noise(clean, noise, timesteps, alpha_bar):
    a = alpha_bar[timesteps].reshape(-1, 1, 1, 1)
    return a.sqrt() * clean + (1 - a).sqrt() * noise


def reconstruct_clean(noisy, noise_prediction, a):
    return (noisy - (1 - a).sqrt() * noise_prediction) / a.sqrt()
```

They implement the same two formulas as above (`a` is \(\bar\alpha\)). I fed the same
numbers to them with `student_checks/evidence/my_hand_check.py` (float64, one
cumulative alpha of 0.81) and kept the output in `my_hand_check.log`:

```text
add_noise           : code 0.2728220211291866   hand 0.2728220211291865   |diff| 5.55e-17
reconstruct_clean   : code 0.4   hand 0.4   |diff| 0.00e+00
MATCH
```

## Conclusion

- Noisy value: hand `0.2728220211291865`, code `0.2728220211291866`
  (difference 5.55e-17, floating-point rounding).
- Recovered clean value: hand `0.4`, code `0.4` (difference 0).

The recovered clean value equals the original clean value exactly. The code in
`add_noise` and `reconstruct_clean` therefore does the same arithmetic as my hand
calculation.
