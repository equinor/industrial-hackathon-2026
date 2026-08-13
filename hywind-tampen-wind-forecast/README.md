# Hywind Tampen Wind Forecast
Predict short-term wind conditions and provide earlier warning of critical wind-drop events.

## Problem

Hywind Tampen is the world's first large-scale floating offshore wind farm supplying power directly
to offshore oil and gas installations. Wind power reduces emissions and fuel consumption, but rapid
weather changes create operational challenges.

Unexpected drops in wind generation can leave operators little time to react. This may lead to
generator overloads, production disruptions, or emergency operational actions. Traditional weather
forecasts often fail to capture the local, short-term wind changes that matter offshore.

In this problem you will investigate whether measurements from surrounding offshore assets
can improve short-term forecasts and warn operators earlier about critical wind drops at Hywind
Tampen.

## What You Could Build

Develop an AI solution that predicts future wind conditions at Hywind Tampen using observations from
nearby offshore installations and the wind farm itself.

Choose one or more objectives:

- Forecast wind conditions for individual turbines over the next 60 minutes. Evaluation metric: root
  mean squared error (RMSE).
- Forecast wind conditions for individual turbines over the next 30 minutes. Evaluation metric: RMSE.
- Estimate forecast uncertainty and confidence intervals. Evaluation metric: pinball loss.

Participants may explore machine learning, deep learning, time-series forecasting, agentic workflows,
and explainable AI techniques.

## Bonus Problem

Build an operator-facing dashboard or intelligent early-warning system. A Wind Drop Alarm should
alert operators when there is a high probability of a significant decrease in wind availability
within the next 30 to 120 minutes.

A useful operator experience should communicate forecast horizon, expected severity, uncertainty,
and recommended action without overstating confidence.

## Data

Participants will receive a preprocessed time-series dataset. See
[data/README.md](data/README.md) for expected locations and data guidance.

### Nearby Offshore Assets

- Statfjord A
- Statfjord B
- Gullfaks C
- Snorre A
- Snorre B
- Visund

### Hywind Tampen Turbines

The dataset covers turbines `HYT-HY01` through `HYT-HY11`.

For each location, wind measurements are represented as:

- U component: east-west wind vector
- V component: north-south wind vector

Using U and V components avoids circular wind-direction data and allows solutions to focus on spatial
and temporal relationships. The locations can be treated as a network of weather sensors where
upwind observations may provide advance information about future conditions at the wind farm.

## Deliverable Considerations

- Specify the forecast target, horizon, sampling interval, and train-validation split.
- Avoid temporal leakage when creating features and evaluation splits.
- Report the required metric per turbine and in aggregate.
- Benchmark against a simple persistence forecast.
- Explain uncertainty calibration and alarm thresholds when applicable.
- Include reproducible setup and run instructions with the solution.
