#!/usr/bin/env python
# coding: utf-8
# Goals:
# Predicting the winner of each match
# Runs scored in each match
# In[ ]:


# Model characteristics(Predicting the world cup winner and runs scored in each match):
# 1.Input two teams data and assuming team1 as winner output 1 - team1 winner , 0- team 2 winner (classification)
# 2.Input two teams data relative to team2 and score prediction (Regression)
# Data Required:
# Squad characteritics(relative to team2): World cup exp, Total Experience, Batting Avg , Bowling Avg, HS, LS etc 
# Team Characteristics(relative to team2): Total win/loss ratio, Win/Loss ratio opp team, Win/Loss ratio at venue, 
# Matches played at venue,Current Ranking, HS, LS, Avg score, Avg score at venue etc
# Training Data: Past fixtures data (Squad characteristics, winner of each match(Target),Runs scored at each match, team current rating, team characteristics)


# In[ ]:


#Features data set columns(Initial)
#Team,Win_loss_ratio_overall(relative), Win/loss ratio opponent, Runs scored overall or batting avg(relative), Runs scored opp, 
#Bowling avg overall(relative), Bowling avg opp, Matches played overall(relative), Home(0 or 1), Matches played at venue(relative)
#Win/loss ratio at venue overall(relative), win/loss ratio at venue opp,Score_margin opp,Score_margin_overall(relative)
#Wickets_margin_overall, Wickets_margin_opp(relative), playing squad score(derived), Win(target variable 0 or 1)


# In[ ]:


#Final fetaures used in training:
#'Rel_squad_Batting_Average', 'Home', 'avg_wicket_overall_Rel',
#'RPO_Overall_Rel', 'fouryear_win_rel', '4year_rpo_rel', 'Home2',
#'3year_team1win_team2'


# In[ ]:


#Final results
#Semis
#England vs Australia
#South Africa vs India
#England wins in semis
#India wins in semis
#Finals
#England vs India -  England wins


# In[310]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[311]:


players=pd.read_csv('Players.csv')


# In[312]:


wcf=pd.read_csv('Cricket_World_cup_fixtures.csv',encoding = "ISO-8859-1")


# In[313]:


wcf.head()


# In[314]:


#teams overall
teams=pd.read_csv('Teams_Overall.csv')


# In[315]:


teams.head()


# In[ ]:





# In[316]:


tfw=pd.read_csv('Teams_Fall_of_Wicket_per_Score.csv')


# In[317]:


tfw.head()


# In[ ]:





# In[318]:


iccm=pd.read_excel("ICC_ODI_Matches.xlsx")


# In[319]:


#stripping trailing leading spaces
iccm['Team1']=iccm['Team1'].str.lstrip()
iccm['Team1']=iccm['Team1'].str.rstrip()
iccm['Team2']=iccm['Team2'].str.lstrip()
iccm['Team2']=iccm['Team2'].str.rstrip()


# In[320]:


#iccm team score data cleaning
def sub(x):
    if(x=='DNB'):
        return pd.Series([0,0])
    if(str(x).find('/')!=-1):
        return pd.Series([x[0:str(x).find('/')],x[str(x).find('/')+1:]])
    return pd.Series([x,10])


# In[227]:


sub('DNB')


# In[321]:



iccm[['Team1_Score','Team2_Wickets']]=iccm['Team1_Score'].apply(lambda x:sub(x) )
iccm[['Team2_Score','Team1_Wickets']]=iccm['Team2_Score'].apply(lambda x:sub(x) )


# In[322]:


iccm.head()


# In[ ]:





# In[323]:


#For math operations
def convertFloat(x):
    y=0
    try:
        y=float(x)
    except:
        y=0
    return y


# In[324]:


iccm['Team1_Score']=iccm['Team1_Score'].apply(lambda x: convertFloat(x)) 
iccm['Team2_Score']=iccm['Team2_Score'].apply(lambda x: convertFloat(x))


# In[ ]:





# In[325]:


players.head()


# In[326]:


#Players dataset cleaning

players.loc[players['Batting_Average']=='-','Batting_Average']=0
players.loc[players['Bowling_Ave']=='-','Bowling_Ave']=0
players.loc[players['Bowling_Strike_Rate']=='-','Bowling_Strike_Rate']=0
players.loc[players['Economy_rate']=='-','Economy_rate']=0
players.loc[players['Batting_Strike_Rate']=='-','Batting_Strike_Rate']=0
players.loc[players['No_of_Wickets']=='-','No_of_Wickets']=0


# In[327]:


players['No_of_Wickets']=players['No_of_Wickets'].astype(dtype=np.float64)
players['No_of_Matches']=players['No_of_Matches'].astype(dtype=np.float64)
players['Batting_Average']=players['Batting_Average'].astype(dtype=np.float64)
players['Bowling_Ave']=players['Bowling_Ave'].astype(dtype=np.float64)
players['Bowling_Strike_Rate']=players['Bowling_Strike_Rate'].astype(dtype=np.float64)
players['Economy_rate']=players['Economy_rate'].astype(dtype=np.float64)
players['Batting_Strike_Rate']=players['Batting_Strike_Rate'].astype(dtype=np.float64)


# In[328]:


#Replacing span column with From and To (Easy for math operations)
players[['From','To']]=players['Span'].apply(lambda x: pd.Series([int(x.split('-')[0]),int(x.split('-')[1])]))


# In[329]:


players['spaninyears']=players['From']-players['To']
#Average player career span estimate give how frequently squad characteristics change 
players['spaninyears'].mean()


# In[516]:


players['Matches_avg_year']=players.apply(lambda x:x['No_of_Matches']/(x['To']-x['From']+1),axis=1)


# In[535]:


playersinsquad('England',1971).sort_values('Matches_avg_year',ascending=False).head(15)['Batting_Average'].mean()


# In[536]:


# Players playing for the country in that particular year (Helps to estimate the probable squad characteristics for that year)
def playersinsquad(country,year):
    if (country in ['Netherlands','Scotland','Ireland','Kenya','Zimbabwe','Bermuda','Canada','East Africa']):
        country='Afghanistan'
        year=2016
    return players[(players['Country']==country) & (players['From']<=year) & (players['To']>=year)].sort_values('Matches_avg_year',ascending=False).head(15)


# In[ ]:





# In[331]:


def playersRelativeScores(team1,team2,year):
    team1s=playersinsquad(team1,year)[['No_of_Matches','No_of_Wickets','Batting_Average','Batting_Strike_Rate','Bowling_Ave','Bowling_Strike_Rate','Economy_rate']].mean()
    team2s=playersinsquad(team2,year)[['No_of_Matches','No_of_Wickets','Batting_Average','Batting_Strike_Rate','Bowling_Ave','Bowling_Strike_Rate','Economy_rate']].mean()
    return team1s-team2s
    


# In[ ]:





# In[537]:


columns=['No_of_Matches','No_of_Wickets','Batting_Average','Batting_Strike_Rate','Bowling_Ave','Bowling_Strike_Rate','Economy_rate']
iccm_columns=['Rel_squad_exp','Rel_squad_wickets','Rel_squad_Batting_Average','Rel_squad_Batting_Strike_Rate','Rel_squad_Bowling_Ave','Rel_squad_Bowling_Strike_Rate','Rel_squad_Economy_rate']
iccm[iccm_columns]=iccm.apply(lambda x:playersRelativeScores(x['Team1'],x['Team2'],x['Match_Date'].year)[columns],axis=1)


# In[333]:


#Test
iccm[60:68]


# In[334]:


#Dropping redundant columns
iccm.drop(['Team2_Played_Overs','Team1_Played_Overs'],axis=1,inplace=True)


# In[335]:


#Changing winner column to 1 or 0 1-Team1 win 0-Team1 loss
iccm.loc[iccm['Winner']==iccm['Team1'],'Winner']=1
iccm.loc[iccm['Winner']==iccm['Team2'],'Winner']=0
iccm['Winner']=iccm['Winner'].astype(dtype=np.float64)


# In[ ]:


#Setting Home (1,0) 1- Team1 home 0-Team away
def home(x):
    if(x):
        return 1
    return 0
iccm['Home']=iccm.apply(lambda x:home(x['Match'].find(x['Team1'])==0),axis=1)


# In[474]:


iccm['Home2']=iccm.apply(lambda x:home(x['Home']==0),axis=1)


# In[337]:


#Dropping some other columns hard to use
iccm.drop(['Margin','Ball_remaining'],axis=1,inplace=True)


# In[338]:


#New columns form teams overall performance
team_columns=['Win_lost_ratio','Ave', 'RPO']
teams['Win_lost_ratio']=teams['Win_lost_ratio'].astype(dtype=np.float64)
teams['Ave']=teams['Ave'].astype(dtype=np.float64)
teams['RPO']=teams['RPO'].astype(dtype=np.float64)


# In[339]:


def setteamdata(column,team):
    val=0
    try:
        val=teams[teams['Team ']==team].iloc[0][column];
    except:
        val=0
    return val
#New columns form teams overall performance
team_columns=['Win_lost_ratio','Ave', 'RPO']
new_columns=['win_loss_ratio_overall_Rel','avg_wicket_overall_Rel','RPO_Overall_Rel']
for idx in range(3):
    iccm[new_columns[idx]]=iccm.apply(lambda x:setteamdata(team_columns[idx],x['Team1'])-setteamdata(team_columns[idx],x['Team2']),axis=1)


# In[ ]:





# In[477]:


#test
iccm.head()


# In[341]:


iccm.drop(['Team1_Inns','Team2_Inns'],axis=1,inplace=True)


# In[ ]:





# In[342]:


#teams past 4 year win rate

def fouryearwinrating(team,year):
    dates=pd.Series([year,year-1]).values
    home_wins=len(iccm[(iccm['Team1']==team) & (iccm['Winner']==1) & (iccm['Match_Date'].dt.year.isin([year-4,year-3,year-2,year-1]))].index)
    away_wins=len(iccm[(iccm['Team2']==team) & (iccm['Winner']==0) & (iccm['Match_Date'].dt.year.isin([year-4,year-3,year-2,year-1]))].index)
    home_lost=len(iccm[(iccm['Team1']==team) & (iccm['Winner']==0) & (iccm['Match_Date'].dt.year.isin([year-4,year-3,year-2,year-1]))].index)
    away_lost=len(iccm[(iccm['Team2']==team) & (iccm['Winner']==1) & (iccm['Match_Date'].dt.year.isin([year-4,year-3,year-2,year-1]))].index)
    ratio=0;
    try:
        ratio=(home_wins + away_wins)/(home_wins + away_wins + home_lost + away_lost)
    except:
        ratio=0
    return ratio


# In[ ]:





# In[ ]:





# In[343]:


#setting columns indicating teams recent performances
iccm['fouryear_win_rel']=iccm.apply(lambda x :fouryearwinrating(x['Team1'],x['Match_Date'].year)-fouryearwinrating(x['Team2'],x['Match_Date'].year),axis=1)


# In[ ]:





# In[344]:


import math
def yearteamscoreavg(team,column,year,n):    
    team1_col='Team1_'+column
    team2_col='Team2_'+column
    span=range(year-n,year,1)
    team1_score=iccm[(iccm['Team1']==team) & (iccm['Match_Date'].dt.year.isin(span))][team1_col]
    team2_score=iccm[(iccm['Team2']==team) & (iccm['Match_Date'].dt.year.isin(span))][team2_col]
    avg=0
    try:
        avg=(team1_score.sum()+team2_score.sum())/(len(team1_score.index)+len(team2_score.index))
    except:
        avg=0
    if(math.isnan(avg)):
        avg=0
    return avg


# In[345]:



yearteamscoreavg('Australia','Score',2019,4)


# In[560]:


def yearwinvsteam(team1,team2,year,n):
    span=range(year-n,year,1)
    home_wins=len(iccm[(iccm['Team1']==team1) & (iccm['Team2']==team2) & (iccm['Winner']==1) & (iccm['Match_Date'].dt.year.isin(span))].index)
    away_wins=len(iccm[(iccm['Team2']==team1) & (iccm['Team1']==team2) & (iccm['Winner']==0) & (iccm['Match_Date'].dt.year.isin(span))].index)
    home_lost=len(iccm[(iccm['Team1']==team1) & (iccm['Team2']==team2) & (iccm['Winner']==0) & (iccm['Match_Date'].dt.year.isin(span))].index)
    away_lost=len(iccm[(iccm['Team2']==team1) & (iccm['Team1']==team2) & (iccm['Winner']==1) & (iccm['Match_Date'].dt.year.isin(span))].index)
    ratio=0;
    try:
        ratio=(home_wins + away_wins)/(home_wins + away_wins + home_lost + away_lost)
        ratio=2*ratio-1
    except:
        ratio=0
    return ratio    


# In[347]:


#setting columns indicating teams recent performances
iccm['4year_score_avg_rel']=iccm.apply(lambda x :yearteamscoreavg(x['Team1'],'Score',x['Match_Date'].year,4)-yearteamscoreavg(x['Team2'],'Score',x['Match_Date'].year,4),axis=1)


# In[651]:


iccm['2year_team1win_team2']=iccm.apply(lambda x:yearwinvsteam(x['Team1'],x['Team2'],x['Match_Date'].year,2),axis=1)


# In[415]:


iccm['4year_team1win_team2']=iccm.apply(lambda x:yearwinvsteam(x['Team1'],x['Team2'],x['Match_Date'].year,4),axis=1)


# In[349]:


iccm['Team1_RPO']=iccm['Team1_RPO'].apply(lambda x:convertFloat(x))
iccm['Team2_RPO']=iccm['Team2_RPO'].apply(lambda x:convertFloat(x))


# In[350]:


#setting columns indicating teams recent performances
iccm['4year_rpo_rel']=iccm.apply(lambda x :yearteamscoreavg(x['Team1'],'RPO',x['Match_Date'].year,4)-yearteamscoreavg(x['Team2'],'RPO',x['Match_Date'].year,4),axis=1)


# In[287]:



teams.columns


# In[546]:


#world cup teams characteristics as of 2018 - insufficient data in 2019
#Squad characteristics 2018 - insufficent data in 2019
columns=['No_of_Matches','No_of_Wickets','Batting_Average','Batting_Strike_Rate','Bowling_Ave','Bowling_Strike_Rate','Economy_rate']
columns_2018=['No_of_Matches_2018','No_of_Wickets_2018','Batting_Average_2018','Batting_Strike_Rate_2018','Bowling_Ave_2018','Bowling_Strike_Rate_2018','Economy_rate_2018']
teams[columns_2018]=teams.apply(lambda x:playersinsquad(x['Team '],2018)[columns].mean(),axis=1)


# In[352]:


teams['score_overall']=teams['Team '].apply(lambda x:teamscoreall(x))


# In[353]:


new_col_2018=['fouryear_score_2019','fouryear_rpo_2019']
teams['fouryear_win_2019']=teams['Team '].apply(lambda x:fouryearwinrating(x,2019))
teams[new_col_2018]=teams['Team '].apply(lambda x:pd.Series([yearteamscoreavg(x,'Score',2019,4),yearteamscoreavg(x,'RPO',2019,4)]))


# In[354]:


teams.columns


# In[355]:


#team characteristics as of 2018
teams.drop(['Span','Won','Lost','Tied','NR','Inns','HS','LS'],axis=1,inplace=True)


# In[356]:


def homewc(team):
    if(team=='England'):
        return 1
    return 0
#setting teams home/away in world cup 2019
teams['Home']=teams['Team '].apply(lambda x:homewc(x))


# In[357]:


teams.head(10)


# In[358]:


iccm.drop(['Match_No','Result','Match'],axis=1,inplace=True)


# In[359]:


iccm.drop(['Team1_Score','Team1_RPO','Team2_Score','Team2_RPO'],axis=1,inplace=True)


# In[ ]:





# In[416]:


iccm.dropna(inplace=True)
iccm.info()


# In[361]:


teams.info()


# In[362]:


iccm.drop(['Team2_Wickets', 'Team1_Wickets'],axis=1,inplace=True)


# In[578]:


corr=iccm.corr()
iccm.corr()


# In[ ]:





# In[539]:


sns.clustermap(corr,cmap = "YlGnBu")


# In[ ]:





# In[366]:


iccm.drop('Ground',inplace=True,axis=1)


# In[368]:


iccm.columns


# In[563]:


sns.pairplot(iccm[['Winner', 'fouryear_win_rel',
       '4year_score_avg_rel', '4year_rpo_rel', '4year_team1win_team2', 'Rel_squad_exp',
       'Rel_squad_wickets','2year_team1win_team2']],hue='Winner')


# In[483]:


iccm['Winner'].value_counts()


# In[ ]:





# In[ ]:





# In[45]:


from sklearn.model_selection import train_test_split


# In[666]:


iccm['Winner']=iccm['Winner'].astype(int)
columnstodrop=['win_loss_ratio_overall_Rel','Winner','Rel_squad_Bowling_Ave','Rel_squad_Bowling_Strike_Rate','Rel_squad_Economy_rate','Rel_squad_exp','4year_score_avg_rel',
              'Rel_squad_Batting_Strike_Rate','Rel_squad_wickets','Team1','Team2','Match_Date','4year_team1win_team2','2year_team1win_team2']
x_train,x_test,y_train,y_test=train_test_split(iccm.drop(columnstodrop,axis=1)[169:],iccm['Winner'][169:],test_size=0.3,random_state=101)


# In[653]:


iccm.columns


# In[374]:


from sklearn.linear_model import LogisticRegression


# In[667]:


logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)


# In[668]:


predictions=logmodel.predict(x_test)


# In[669]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[ ]:





# In[670]:


logmodel.coef_


# In[672]:


# model
x_train.columns


# In[487]:


#Current data set preparation from fixtures
wcf.head(10)


# In[506]:


wcf.info()


# In[679]:


prediction_col=['Rel_squad_Batting_Average', 'Home',
       'avg_wicket_overall_Rel', 'RPO_Overall_Rel', 'fouryear_win_rel','4year_rpo_rel','Home2','3year_team1win_team2']
teams_col=['Batting_Average_2018','Home','Ave','RPO','fouryear_win_2019','fouryear_rpo_2019','No_of_Matches_2018','Mat']
len(teams_col)


# In[394]:


teams.columns


# In[583]:


teams


# In[525]:


wcf.columns


# In[447]:


wcf['Team1']=wcf['Team1'].str.lstrip()
wcf['Team1']=wcf['Team1'].str.rstrip()
wcf['Team2']=wcf['Team2'].str.lstrip()
wcf['Team2']=wcf['Team2'].str.rstrip()


# In[680]:


x_predict=pd.DataFrame()
x_predict[prediction_col]=wcf[0:45].apply(lambda x:teams[teams['Team ']==x['Team1']][teams_col].iloc[0]-teams[teams['Team ']==x['Team2']][teams_col].iloc[0],axis=1)


# In[486]:


def homewcf(team):
    if(team=='England'):
        return 1
    return 0    


# In[681]:


x_predict['Home']=wcfleague.apply(lambda x:homewcf(x['Team1']),axis=1 )


# In[682]:


x_predict['Home2']=wcfleague.apply(lambda x:homewcf(x['Team2']),axis=1 )


# In[683]:


x_predict['3year_team1win_team2']=wcf[0:45].apply(lambda x:yearwinvsteam(x['Team1'],x['Team2'],2019,3),axis=1)


# In[684]:


x_predict.head(10)


# In[685]:


yearwinvsteam('England','South Africa',2019,3)


# In[ ]:





# In[469]:


teams.head(10)


# In[686]:


y_predict=logmodel.predict(x_predict)


# In[687]:


def getwinner(x):
    if(x['Winner']==1):
        return x['Team1']
    return x['Team2']


# In[688]:


wcfleague=wcf[0:45]


# In[689]:


wcfleague['Winner']=pd.Series(y_predict)


# In[690]:


wcfleague.head(10)


# In[691]:


wcfleague['Winner']=wcfleague.apply(lambda x:getwinner(x),axis=1)


# In[695]:


wcfleague[0:10]


# In[704]:


wcfleague[20:30]


# In[703]:


wcfleague[30:40]


# In[693]:


wcfleague['Winner'].value_counts()


# In[587]:


#England, South Africa, India, Australia qualify for semis


# In[61]:


#Semis
#England vs Australia
#South Africa vs India


# In[ ]:


#According to table
#England wins in semis
#India wins in semis


# In[595]:


#Finals
#England vs India -  England wins


# In[ ]:




