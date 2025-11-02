# CS-230-Deep-Learning-Project
This repository was created for the final CS230 project, focused on predicting electricity market pricing. 
## Introduction 

With a growth in electricification of heavily data-reliant business models, grid dispatching, transmission and reliability have now become the cornerstone of the current electrified economy. In this specific sector, Electricity Price Forecasting (EPF) plays an increasing role especially as day-ahead and real-time price forecasting supports better bidding strategies for producers, improves demand response for consumers, and aids grid operators in dispatch decisions.\
As recent technologies have made renewable energies ubiquitous, their penetration in today's power systems make price preidictions more uncertain and volatile. With those new constrains, EPF constantly provide new tools with the ultimate objective of narrowing the gap between predictions and actual prices. 

## Goal and Scope

The goal of this project is to build an efficient deep learning model able to predict a day-ahead hourly price distribution, given past prices, and forecasted loads, renewables generations and calendar or seasonal constraints (weekdays vs weekends or Winter vs Summer).The scope will aim at creating three different time-sequence Deep Learning models (baseline, Recurrent NN/Convolutional NN based on LSTM and Tranformer) and comparing their performance with state-of-art models from the [epftoolbox](https://github.com/jeslago/epftoolbox)


## Notes

# Data : 
- I downloaded historical data from 2 US Markets (NYISO and CAISO) consisting in an hourly renewables_mix (Wind, Hydro, Geothermal, Solar, Other), hourly load and hourly LMP day ahead price (averaged over all present locations). They all have only the last 39 months of data so I started downloading from Jan 2020 to today. I haved the package [gridstatus](https://opensource.gridstatus.io/en/latest/installation.html)
- Can to the sam with the ENTSO-E European market (but need an API key ?)
- One thing can be showing that adding exogeneous variables allows to a better prediction of the future LMP price
- Refine this accuracy on different modle architectures (MLP, LSTM, Transformer)
- Provide necessary justification for train/validate/test split


## Litterature review  
- [Paper 1](https://www.sciencedirect.com/science/article/pii/S0306261921004529) 
State-of-the-art algorithms for used for electricity price forecasting, also provides the [epftoolbox](https://github.com/jeslago/epftoolbox) which is designed to facilitate reproducible research in electricity price forecasting.
- [Paper 2](https://arxiv.org/abs/2008.08006)
Comprehensive comparison of two most common structures when using the deep neural networks -- one that focuses on each hour of the day separately, and one that reflects the daily auction structure and models vectors of the prices.
- [Paper 3](https://arxiv.org/abs/2104.05522)
Forecasting electricity prices with an augmented model including exogeneous variables
- [Paper 4](https://arxiv.org/abs/2508.04875)
Foundation model that integrates graph-based inductive biases to capture spatial interdependencies across interconnected electricity markets in European regions
- [Paper 5](https://www.sciencedirect.com/science/article/pii/S0360544224026513)
Deep learning-based approach with a Multilayer Perceptron across European markets
- [Paper 6](https://arxiv.org/abs/2308.01436)
Emphasis “price-aware” learning, i.e., embedding system features that reflect how prices form (supply bids, demand, transmission limits). Instead of treating price as just a time series, they explicitly incorporate network congestion, load levels, and renewable generation into the model.
- [Paper 7](https://arxiv.org/pdf/1808.05527)
The first paper about DL for electricity markets which introduces LSTM models on price time sequences
- [Paper 8] (https://www.sciencedirect.com/science/article/pii/S030626191830196X?fr=RR-2&ref=pdf_download&rr=98c78e308fe3eb32)
Comparison of emeprical algorithms for EPF via Deep Learning apporaches
- [Paper 9] (https://ieeexplore.ieee.org/document/10258431)
Machine Learning techniques applied to the NYISO for a physics-aware price predicition


