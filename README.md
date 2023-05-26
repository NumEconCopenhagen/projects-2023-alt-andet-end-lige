# Alt andet end lige

This repository contains  
1. Inaugural project. 
- In this project, we numerically solve the model from the assignment "Time Use of Couples". The project explores the division of labor in the home and in the market between males and females and how this work division impacts the aggregate utility of the household.

2. Data project. 
- We fetch data from **Statistics Denmark** on **prices of housing** and the **unemployment rate** across provinces in Denmark using the API from Statistics Denmark. After cleaning the data, we analyze the relationship between the unemployment rate and the price of housing across provinces. 

3. Model project. 
- We numerically solve the Principal-Agent model with adverse selection. Initially we let the principal offer two kinds of workers with different productivity the same contract. We then allow for signalling of productivity through conditioning on the number of years of education and analyze the implications for the model. Lastly, we allow for *n* types of workers and *n* types of contracts and analyze the implications of this extension.

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires no further packages.

**Folder structure:** 
- Each project is contained in its own folder. The solution to each project consists of a notebook (from which the project is run), a py-file with classes and functions, and a readme-file. All four projects should be run from the notebooks in their respective folders. 
