# CryptoMarketForecast
This is a machine learning forecasting framework with historical cryptocurrency market data retrieval from Binance API.

![04307e1c-c947-4081-b38d-12d8341a4985](https://user-images.githubusercontent.com/38635941/136774875-342aabf9-a4d7-48e5-8c35-e351efd032d4.jpg)
### Overview:
This is a complete system entirely developed in python developed as a master thesis in Instituto Superior Técnico (University of Lisbon, Portugal).
This system contains all steps prior to and of the forecasting process:
1. Data downloading and storing through Binance's API;
2. Data cleansing and preprocessing (developed an untested resampling process, read more about this step in the links below containing the documentation);
3. Machine Learning training (Ensemble of 4 algorithms Logistic Regression, Random Forest, XGBoost & Support Vector Classifier) and testing;
4. Data Visualization (mainly using the Bokeh lib).

The core file is the obviously named **main.py** file. 

To read the full thesis and documentation you can use any of the following links (only the last one is available free of charge, but please cite the work in the first link):
- [Science Direct](https://www.sciencedirect.com/science/article/abs/pii/S1568494620301277);
- [Springer Books](https://www.springer.com/gp/book/9783030683788?utm_campaign=3_pier05_buy_print&utm_content=en_08082017&utm_medium=referral&utm_source=google_books#otherversion=9783030683795);
- [Instituto Superior Técnico (Free)](https://fenix.tecnico.ulisboa.pt/cursos/meec/dissertacao/1128253548921836).

**Note:** If you find my work valuable and worth citing, please include the paper from **Science Direct** in your citations.

