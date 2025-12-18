################################################################################
#####                            Introduction                              #####
################################################################################
# This code is a part of a project. The aim of the project is to predict       #
# whether a student will receive a 'Good' GPA based on a number of predictors. #
# I will be processing and analysising data taken from the dataset             #
# "📚 Students Performance Dataset 📚" by [Rabie El Kharoua], licensed under   #
# CC-BY 4.0 https://creativecommons.org/licenses/by/4.0/.                      #
# DOI citation: Rabie El Kharoua. (2024). 📚 Students Performance Dataset 📚   #
#               [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/5195702 #
# The code will include brief comments, in order to clarify what it does, but  # 
# is intended to be read in conjunction with the written report.               #
# Both the code and comments have been made to  fit a line width of no more    #
# than 80 characters, this is done to allow the viewing of the code, comments  #
# and graphs at the same time when the code is being run.                      #                                     # 
# The version of R used is: 4.5.2                                              #
################################################################################

################################################################################
#####                             Preliminaries                            #####
################################################################################

##### Align R studio version to the one used when the code was written #####
R.version.string
 # When running the code you will want to make sure your R version is 4.5.2.
 # If the version does not match packages may not load correctly and or code
 # may not work, please make sure to install the correct version before 
 # continuing.

##### Install necessary packages #####
install.packages("tidyverse") # Package version '2.0.0'
install.packages("janitor") # Package version '2.2.1'
install.packages("MASS") # Package version '7.3.65'
install.packages("glmnet") # Package version '4.1.10'
install.packages("hnp") # Package version '1.2.7'
install.packages("keras3") # Package version '1.4.0'
install.packages("randomForest") # Package version '4.7.1.2'
install.packages("e1071") # Package version '1.7.16'
install.packages("tensorflow") # Package version '2.20.0'
##### Load necessary packages #####
library(tidyverse)
library(janitor)
library(MASS)
library(glmnet)
library(dplyr)
library(hnp)
library(keras3)
library(randomForest)
library(e1071)
library(tensorflow)


# Before moving on make sure you have the data set downloaded into a known     #
# location, the file must be in csv layout. You then need to set your working  #
# directory said location. In order to do this click the 'Session' option on   #
# the menu bar, then select 'Set Working Directory' and 'Choose Directory'     #
# from here choose the folder where the data file is stored.                   #
# I suggest saving the file as 'student_performance'.                          #

################################################################################
#####                             Data Input                               #####
################################################################################
df <- read.csv("student_performance.csv", header = T)
 # This code reads the data file into R and stores it as an object called
 # 'df'.
 # If you choose to save the data set under a different name please replace 
 # "student_performance.csv" with the appropriate name. 


################################################################################
#####                      Exploratory Data Analysis                       #####
################################################################################

##### Introduction to data set ######

dim(df)
 # This code prints two values, the value on the left is the amount of rows the
 # data set contains, the value on the right is the amount of columns the data
 # set contains. 
 # The data set has 2392 rows and 15 columns. 

colnames(df)
 # This code shows us all the column names. 
 # the data set has the following column names: 'StudentID', 'Age', 'Gender', 
 # 'Ethnicity', ' ParentalEducation', 'StudyTimeWeekly', 'Absences',
 # 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music',
 # 'Volunteering', 'GPA' and 'GradeClass'.



################################################################################
#####                         Data Quality Check                           #####
################################################################################

##### Completeness #####

sum(is.na(df))
 # This prints the amount of missing values within the data set.
 # Printed '0' indicating there are no NA's. 

##### Uniqueness #####

any(duplicated(df))
 # This code checks for any rows where all the variables are a repeat of those
 # of a previous row.
 # This prints 'False' meaning there are no repeated rows.

any(duplicated(t(df)))
 # This code checks for any columns where all the variables are a repeat of those
 # of a previous row.
 # This prints 'False' meaning there are no repeated columns.

any(duplicated(df$StudentID))
 # This code checks to make sure each 'StudentID' is unique.
 # This prints 'False' meaning there are no repeated ID's.

##### Validity #####

any(!df$Age %in% c(15,16,17,18))
any(!df$Gender %in% c(0,1))
any(!df$Ethnicity %in% c(0,1,2,3))
any(!df$ParentalEducation %in% c(0,1,2,3,4))
any(df$StudyTimeWeekly < 0)
any(!(df$Absences >= 0 & df$Absences <=190))
any(!df$Tutoring %in% c(0,1))
any(!df$ParentalSupport %in% c(0,1,2,3,4))
any(!df$Extracurricular %in% c(0,1))
any(!df$Sports %in% c(0,1))
any(!df$Music %in% c(0,1))
any(!df$Volunteering %in% c(0,1))
any(!(df$GPA >= 0 & df$GPA <= 4.0))
any(!df$GradeClass %in% c(0,1,2,3,4))
 # This chunk of code checks to make sure our values are all within the range
 # we expect them to be. 
 # Each of these print 'FALSE' meaning they are all within their expected 
 # ranges.

df$temp <- ifelse(df$GPA >= 3.5, 0, 
                  ifelse(df$GPA >= 3.0, 1, 
                         ifelse(df$GPA >= 2.5, 2, 
                                ifelse(df$GPA >=2.0, 3, 4))))
 # This creates another column within the data set called 'temp', each row has
 # a number , 0,1,2,3,4 , based upon their GPA. Using the same levels for each
 # as those which should be used for 'GradeClass'. 

any(!(df$GradeClass == df$temp))
 # This code will tell us if there are any rows where its 'GradeClass' does not
 # equal its 'temp'. Indicating that the 'GPA' and 'GradeClass' do not align.
 # This prints 'TRUE' meaning there are such rows within our data set.
sum(!(df$GradeClass == df$temp))
 # This code tells us the amount of rows where 'GradeClass' does not align with
 # temp. 
 # We get 168 such rows. 

##### Consistency #####

str(df)
 # This allows us to see what data type each variable is. As we can see 
 # 'StudentID', 'Age', 'Gender', 'Ethnicity', ' ParentalEducation', 'Absences',
 # 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 
 # 'Volunteering' and 'GradeClass' are integers. 'StudyTimeWeekly' and 'GPA' are
 # numerical.
 # There is no need to change these.

table(df$Gender)
table(df$Ethnicity)
table(df$ParentalEducation)
table(df$Tutoring)
table(df$ParentalSupport)
table(df$Extracurricular)
table(df$Sports)
table(df$Music)
table(df$Volunteering)
table(df$GradeClass)
 # This chunk of code creates a table for each column needed. It lists the count
 # for each category in the variable. 
 # No category has little to no data, so we do not need to make an altercations.

##### Outliers #####

ggplot(df, aes(x = Age)) + geom_boxplot()
 # This creates a boxplot of the data for 'Age'.
 # From this graph we can see that there are no outliers for 'Age'.

ggplot(df, aes(x = StudyTimeWeekly)) + geom_boxplot()
 # This creates a boxplot of the data for 'StudyTimeWeekly'.
 # From this graph we can see that there are no outliers for 'StudyTimeWeekly'.

ggplot(df, aes(x = Absences)) + geom_boxplot()
# This creates a boxplot of the data for 'Absences'.
# From this graph we can see that there are no outliers for 'Absences'.

##### Data Cleaning #####

df <- df %>% filter(df$GradeClass == df$temp)
 # This removes any rows where 'GradeClass' and 'temp' do not align.

df <- df %>% dplyr::select(-Age)
df <- df %>% dplyr::select(-StudentID)
df <- df %>% dplyr::select(-GPA)
df_clean <- df %>% dplyr::select(-temp)
 # This chunk of code removes all the columns we do not need for our project. 
 # It also renames 'df' to 'df_clean' this is our cleaned data set.

df_clean$GradeClass <- ifelse(df_clean$GradeClass <= 1, 1, 0)
 # This code changes the GradeClass column into a binary classification. Those
 # rows with a 'GradeClass' value of 2,3 or 4 have been assigned 0 and those
 # rows with a 'GradeClass' value of 0,1 have been assigned 1.

df_clean$Gender <- ifelse(df_clean$Gender <= 0, "Male", "Female")
df_clean$Ethnicity <- ifelse(df_clean$Ethnicity <= 0, "A", 
                  ifelse(df_clean$Ethnicity <= 1, "B", 
                         ifelse(df_clean$Ethnicity <= 2, "C","D"))) 
df_clean$ParentalEducation <- ifelse(df_clean$ParentalEducation <= 0, "A", 
                              ifelse(df_clean$ParentalEducation <= 1, "B", 
                              ifelse(df_clean$ParentalEducation <= 2, "C", 
                              ifelse(df_clean$ParentalEducation <=3, "D",                                        "E"))))
df_clean$Tutoring <- ifelse(df_clean$Tutoring <= 0, "No", "Yes")
df_clean$ParentalSupport <- ifelse(df_clean$ParentalSupport <= 0, "A", 
                            ifelse(df_clean$ParentalSupport <= 1, "B", 
                            ifelse(df_clean$ParentalSupport <= 2, "C", 
                            ifelse(df_clean$ParentalSupport <=3, "D","E"))))
df_clean$Extracurricular <- ifelse(df_clean$Extracurricular <= 0, "No", "Yes")
df_clean$Sports <- ifelse(df_clean$Sports <= 0, "No", "Yes")
df_clean$Music <- ifelse(df_clean$Music <= 0, "No", "Yes")
df_clean$Volunteering <- ifelse(df_clean$Volunteering <= 0, "No", "Yes")
 # This chunk of code changes any categorical predictor variables which are in
 # numerical format to characters. This allows r to read them as categories.
                                
################################################################################
#####           Background Information and Initial Data Overview           #####
################################################################################

# We will now begin working with our cleaned data set #

dim(df_clean)
 # This code tells us how many rows and columns our data set has.

head(df_clean)
 # This prints the first few rows of the data set it allows us to get a general 
 # idea for the data set and we can also see if it loaded in correctly. 

str(df_clean)
 # This allows us to see what data type each variable is.

################################################################################
#####                         Univariate Analysis                          #####
################################################################################

##### Variable 1: 'GradeClass' #####
tabyl(df_clean$GradeClass)
 # This creates a frequency table for 'GradeClass' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; 0 - 1924, 1 - 300.
 # As a percentage that is; 0 - 86.5%, 1 - 13.5%.

ggplot(df_clean, aes(x=factor(GradeClass))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 2: 'Gender' #####
tabyl(df_clean$Gender)
 # This creates a frequency table for 'Gender' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; Male - 1081, Female - 1143.
 # As a percentage that is; Male - 48.6%, Female - 51.4%.

ggplot(df_clean, aes(x=factor(Gender))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 3: 'Ethnicity' #####
tabyl(df_clean$Ethnicity)
 # This creates a frequency table for 'Ethnicity' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; A - 1127, B - 465, C - 428, D - 204.
 # As a percentage that is; A - 50.7%, B - 20.9%, C - 19.2%, D - 9.2%.

ggplot(df_clean, aes(x=factor(Ethnicity))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 4: 'ParentalEducation' #####
tabyl(df_clean$ParentalEducation)
 # This creates a frequency table for 'ParentalEducation' allowing us to see its
 # distribution, it includes the count and percentage of each outcome.
 # The count are also follows; A - 226, B - 682, C - 864, D - 335, E - 117.
 # As a percentage that is; A - 10.2%, B - 30.7%, C - 38.8%, D - 15.1%, E - 5.2%

ggplot(df_clean, aes(x=factor(ParentalEducation))) + geom_bar() + 
  theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 5: 'StudyTimeWeekly' #####
summary(df_clean$StudyTimeWeekly)
 # This code shows the following statistical information for 'StudyTimeWeekly';
 # Minimum, 1st Quantile, Median, Mean, 3rd Quantile and Maximum.
 # The entries in the 'StudyTimeWeekly' column range from 0.001 to 19.98, with a
 # median of 9.72 and a mean of 9.79.

ggplot(df_clean, aes(x=StudyTimeWeekly)) + geom_histogram(binwidth = 0.5,) + 
  theme_minimal()
 # This plots a histogram plot for 'StudyTimeWeekly', it represents the
 # frequency count of age in bins, this has a binwidth of 0.5 meaning each 
 # bar represents a half hour of study time. 
 # The count for each bin does not differ greatly. 


##### Variable 6: 'Absences' #####
summary(df_clean$Absences)
 # This code shows the following statistical information for 'Absences'; 
 # Minimum, 1st Quantile, Median, Mean, 3rd Quantile and Maximum.
 # The entries in the 'Minimum' column range from 0 to 29 , with a median of 15
 # and a mean of 14.55.

ggplot(df_clean, aes(x=Absences)) + geom_histogram(binwidth = 1,) + 
  theme_minimal()
 # This plots a histogram plot for 'Absences', it represents the
 # frequency count of age in bins, this has a binwidth of 1 meaning each 
 # bar represents one absence. 
 # The count for each bin does not differ greatly. 


##### Variable 7: 'Tutoring' #####
tabyl(df_clean$Tutoring)
 # This creates a frequency table for 'Tutoring' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; No - 1552, Yes - 672.
 # As a percentage that is; No - 69.8%, Yes - 30.2%.

ggplot(df_clean, aes(x=factor(Tutoring))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 8: 'ParentalSupport' #####
tabyl(df_clean$ParentalSupport)
 # This creates a frequency table for 'ParentalSupport' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; A - 196, B - 460, C - 692, D - 643, E - 233.
 # As a percentage that is; A - 8.8%, B - 20.7%, C - 31.1%, D - 28.9%, E - 10.5%

ggplot(df_clean, aes(x=factor(ParentalSupport))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 9: 'Extracurricular' #####
tabyl(df_clean$Extracurricular)
 # This creates a frequency table for 'Extracurricular' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; No - 1366, Yes - 858.
 # As a percentage that is; No - 61.4%, Yes - 38.6%.

ggplot(df_clean, aes(x=factor(Extracurricular))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 10: 'Sports' #####
tabyl(df_clean$Sports)
 # This creates a frequency table for 'Sports' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; No - 1550, Yes - 674.
 # As a percentage that is; No - 69.7%, Yes - 30.3%.

ggplot(df_clean, aes(x=factor(Sports))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.


##### Variable 11: 'Music' #####
tabyl(df_clean$Music)
# This creates a frequency table for 'Music' allowing us to see its
# distribution, it includes the count and percentage of each outcome.
# The count are also follows; No - 1778, Yes - 446.
# As a percentage that is; No - 79.9%, Yes - 20.1%.

ggplot(df_clean, aes(x=factor(Music))) + geom_bar() + theme_minimal()
# This code plots a bar chart for the counts above, providing a visual.


##### Variable 12: 'Volunteering' #####
tabyl(df_clean$Volunteering)
 # This creates a frequency table for 'Volunteering' allowing us to see its
 # distribution, it includes the count and percentage of each outcome. 
 # The count are also follows; No - 1876, Yes - 348.
 # As a percentage that is; No - 84.4%, Yes - 15.6%.

ggplot(df_clean, aes(x=factor(Volunteering))) + geom_bar() + theme_minimal()
 # This code plots a bar chart for the counts above, providing a visual.



################################################################################
#####                         Bivariate Analysis                           #####
################################################################################

##### Dependent variable (GradeClass) vs Independent variable #####

### Gender ###
table(df_clean$Gender, df_clean$GradeClass)
 # This code creates a contingency table for 'Gender' and 'GradeClass'.
prop.table(table(df_clean$Gender, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see the percentage of Female and Male student who do have 
 # a Good GPA is even with both being around 13%.

summary(glm(GradeClass ~ Gender, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Gender as the exploratory variable, it then prints a detailed
 # summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### Ethnicity ###
table(df_clean$Ethnicity, df_clean$GradeClass)
 # This code creates a contingency table for 'Ethnicity' and 'GradeClass'.
prop.table(table(df_clean$Ethnicity, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see the percentage that the percentage of students who 
 # did achieve a Good GPA is relatively the same despite ethnicity range from 
 # 12.7% to 15.3%.

summary(glm(GradeClass ~ Ethnicity, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Ethnicity as the exploratory variable, it then prints a detailed
 # summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### ParentalEducation ###
table(df_clean$ParentalEducation, df_clean$GradeClass)
 # This code creates a contingency table for 'ParentalEducation' and 
 # 'GradeClass'.
prop.table(table(df_clean$ParentalEducation, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see that there only is a slight difference for those whose
 # parents have a 'Higher' education with them having a smaller percentage of
 # those with a Good GPA of around 9% compared to around 13%.

summary(glm(GradeClass ~ ParentalEducation, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and ParentalEducation as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### StudyTimeWeekly ###
ggplot(df_clean, aes(y=StudyTimeWeekly, x = factor(GradeClass))) + 
  theme_minimal() + geom_boxplot()
 # This creates a box plot representing the distribution of StudyTimeWeekly
 # for those who have a GradeClass '0' and those who have a grade class '1'.

summary(glm(GradeClass ~ StudyTimeWeekly, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and StudyTimeWeekly as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### Absences ### 
ggplot(df_clean, aes(y=Absences, x = factor(GradeClass))) + theme_minimal() + 
  geom_boxplot()
 # This creates a box plot representing the distribution of Absences for those
 # who have a GradeClass '0' and those who have a grade class '1'.

summary(glm(GradeClass ~ Absences, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Absences as the exploratory variable, it then prints a detailed
 # summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### Tutoring ###
table(df_clean$Tutoring, df_clean$GradeClass)
 # This code creates a contingency table for 'Tutoring' and 'GradeClass'.
prop.table(table(df_clean$Tutoring, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see those who receive tutoring are more likely to obtain a 
 # Good GPA at 20.8% as compared to 10.3% for those who didn't.

summary(glm(GradeClass ~ Tutoring, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Tutoring as the exploratory variable, it then prints a detailed
 # summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### ParentalSupport ###
table(df_clean$ParentalSupport, df_clean$GradeClass)
 # This code creates a contingency table for 'ParentalSupport' and 'GradeClass'.
prop.table(table(df_clean$ParentalSupport, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see that as parental support increase so does the percentage 
 # of those who achieve a Good GPA. Showing a positive correlation. 

summary(glm(GradeClass ~ ParentalSupport, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and ParentalSupport as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### Extracurricular ###
table(df_clean$Extracurricular, df_clean$GradeClass)
 # This code creates a contingency table for 'Extracurricular' and 'GradeClass'.
prop.table(table(df_clean$Extracurricular, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see that the percentage of those who achieve a Good GPA is 
 # slightly higher at 16.4% for those who do participate in an extracurricular
 # activity as opposed to those who don't at 11.6%.

summary(glm(GradeClass ~ Extracurricular, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Extracurricular as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 

### Sports ###
table(df_clean$Sports, df_clean$GradeClass)
 # This code creates a contingency table for 'Sports' and 'GradeClass'.
prop.table(table(df_clean$Sports, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see that the percentage of those who achieve a Good GPA is 
 # slightly higher for students who do participate in a sport compared to those 
 # who don't at 16.6% versus 12.1%

summary(glm(GradeClass ~ Sports, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Sports as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 

### Music ###
table(df_clean$Music, df_clean$GradeClass)
 # This code creates a contingency table for 'Music' and 'GradeClass'.
prop.table(table(df_clean$Music, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see that the percentage of those who achieve a Good GPA is 
 # slightly higher for students who do participate in music activities as 
 # opposed to those who don't at 17.9% compared to 12.4%.

summary(glm(GradeClass ~ Music, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Music as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


### Volunteering ###
table(df_clean$Volunteering, df_clean$GradeClass)
 # This code creates a contingency table for 'Volunteering' and 'GradeClass'.
prop.table(table(df_clean$Volunteering, df_clean$GradeClass), 1)
 # This creates a probability table showing the proportions of the contingency
 # table.
 # From this we can see that the percentage of those who achieve a Good GPA is 
 # around the same for both those who do and don't volunteer, with values 
 # reading in at 13.4% and 14.1%.

summary(glm(GradeClass ~ Volunteering, data = df_clean, family = binomial))
 # This code fits a logistic regression model with GradeClass as the response
 # variable and Volunteering as the exploratory variable, it then prints a 
 # detailed summary of said fitted logistic regression model. 
 # This will explored further in the written report. 


##### Independent variable vs Independent variable #####

### Categorical vs Categorical ###

table(df_clean$Gender, df_clean$Ethnicity)
 # This code creates a contingency table regarding Gender and Ethnicity. 
chisq.test(table(df_clean$Gender, df_clean$Ethnicity))
 # This code runs a chi-sq test for Gender and Ethnicity.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$ParentalEducation)
 # This code creates a contingency table regarding Gender and 
 # ParentalEducation. 
chisq.test(table(df_clean$Gender, df_clean$ParentalEducation))
 # This code runs a chi-sq test for Gender and ParentalEducation.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$Tutoring)
 # This code creates a contingency table regarding Gender and Tutoring. 
chisq.test(table(df_clean$Gender, df_clean$Tutoring))
 # This code runs a chi-sq test for Gender and Tutoring.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$ParentalSupport)
 # This code creates a contingency table regarding Gender and 
 # ParentalSupport. 
chisq.test(table(df_clean$Gender, df_clean$ParentalSupport))
 # This code runs a chi-sq test for Gender and ParentalSupport.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$Extracurricular)
 # This code creates a contingency table regarding Gender and 
 # Extracurricular. 
chisq.test(table(df_clean$Gender, df_clean$Extracurricular))
 # This code runs a chi-sq test for Gender and Extracurricular.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$Sports)
 # This code creates a contingency table regarding Gender and Sports.
chisq.test(table(df_clean$Gender, df_clean$Sports))
 # This code runs a chi-sq test for Gender and Sports.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$Music)
 # This code creates a contingency table regarding Gender and Music.
chisq.test(table(df_clean$Gender, df_clean$Music))
 # This code runs a chi-sq test for Gender and Music.
 # This will be explored further in the written report.

table(df_clean$Gender, df_clean$Volunteering)
 # This code creates a contingency table regarding Gender and Volunteering.
chisq.test(table(df_clean$Gender, df_clean$Volunteering))
 # This code runs a chi-sq test for Gender and Volunteering.
 # This will be explored further in the written report.


table(df_clean$Ethnicity, df_clean$ParentalEducation)
 # This code creates a contingency table regarding Ethnicity and 
 # ParentalEducation. 
chisq.test(table(df_clean$Ethnicity, df_clean$ParentalEducation))
 # This code runs a chi-sq test for Ethnicity and ParentalEducation.
 # This will be explored further in the written report.

table(df_clean$Ethnicity, df_clean$Tutoring)
 # This code creates a contingency table regarding Ethnicity and Tutoring. 
chisq.test(table(df_clean$Ethnicity, df_clean$Tutoring))
 # This code runs a chi-sq test for Ethnicity and Tutoring.
 # This will be explored further in the written report.

table(df_clean$Ethnicity, df_clean$ParentalSupport)
 # This code creates a contingency table regarding Ethnicity and 
 # ParentalSupport. 
chisq.test(table(df_clean$Ethnicity, df_clean$ParentalSupport))
 # This code runs a chi-sq test for Ethnicity and ParentalSupport.
 # This will be explored further in the written report.

table(df_clean$Ethnicity, df_clean$Extracurricular)
 # This code creates a contingency table regarding Ethnicity and 
 # Extracurricular. 
chisq.test(table(df_clean$Ethnicity, df_clean$Extracurricular))
 # This code runs a chi-sq test for Ethnicity and Extracurricular.
 # This will be explored further in the written report.

table(df_clean$Ethnicity, df_clean$Sports)
 # This code creates a contingency table regarding Ethnicity and Sports.
chisq.test(table(df_clean$Ethnicity, df_clean$Sports))
 # This code runs a chi-sq test for Ethnicity and Sports.
 # This will be explored further in the written report.

table(df_clean$Ethnicity, df_clean$Music)
 # This code creates a contingency table regarding Ethnicity and Music.
chisq.test(table(df_clean$Ethnicity, df_clean$Music))
 # This code runs a chi-sq test for Ethnicity and Music.
 # This will be explored further in the written report.

table(df_clean$Ethnicity, df_clean$Volunteering)
 # This code creates a contingency table regarding Ethnicity and 
 # Volunteering.
chisq.test(table(df_clean$Ethnicity, df_clean$Volunteering))
 # This code runs a chi-sq test for Ethnicity and Volunteering.
 # This will be explored further in the written report.


table(df_clean$ParentalEducation, df_clean$Tutoring)
 # This code creates a contingency table regarding ParentalEducation and 
 # Tutoring. 
chisq.test(table(df_clean$ParentalEducation, df_clean$Tutoring))
# This code runs a chi-sq test for ParentalEducation and Tutoring.
# This will be explored further in the written report.

table(df_clean$ParentalEducation, df_clean$ParentalSupport)
 # This code creates a contingency table regarding ParentalEducation and 
 # ParentalSupport. 
chisq.test(table(df_clean$ParentalEducation, df_clean$ParentalSupport))
 # This code runs a chi-sq test for ParentalEducation and ParentalSupport.
 # This will be explored further in the written report.

table(df_clean$ParentalEducation, df_clean$Extracurricular)
 # This code creates a contingency table regarding ParentalEducation and 
 # Extracurricular. 
chisq.test(table(df_clean$ParentalEducation, df_clean$Extracurricular))
 # This code runs a chi-sq test for ParentalEducation and Extracurricular.
 # This will be explored further in the written report.

table(df_clean$ParentalEducation, df_clean$Sports)
 # This code creates a contingency table regarding ParentalEducation and 
 # Sports.
chisq.test(table(df_clean$ParentalEducation, df_clean$Sports))
 # This code runs a chi-sq test for ParentalEducation and Sports.
 # This will be explored further in the written report.

table(df_clean$ParentalEducation, df_clean$Music)
 # This code creates a contingency table regarding ParentalEducation and 
 # Music.
chisq.test(table(df_clean$ParentalEducation, df_clean$Music))
 # This code runs a chi-sq test for ParentalEducation and Music.
 # This will be explored further in the written report.

table(df_clean$ParentalEducation, df_clean$Volunteering)
 # This code creates a contingency table regarding ParentalEducation and
 # Volunteering.
chisq.test(table(df_clean$ParentalEducation, df_clean$Volunteering))
 # This code runs a chi-sq test for ParentalEducation and Volunteering.
 # This will be explored further in the written report.


table(df_clean$Tutoring, df_clean$ParentalSupport)
 # This code creates a contingency table regarding Tutoring and 
 # ParentalSupport. 
chisq.test(table(df_clean$Tutoring, df_clean$ParentalSupport))
 # This code runs a chi-sq test for Tutoring and ParentalSupport.
 # This will be explored further in the written report.

table(df_clean$Tutoring, df_clean$Extracurricular)
 # This code creates a contingency table regarding Tutoring and 
 # Extracurricular. 
chisq.test(table(df_clean$Tutoring, df_clean$Extracurricular))
 # This code runs a chi-sq test for Tutoring and Extracurricular.
 # This will be explored further in the written report.

table(df_clean$Tutoring, df_clean$Sports)
 # This code creates a contingency table regarding Tutoring and Sports.
chisq.test(table(df_clean$Tutoring, df_clean$Sports))
 # This code runs a chi-sq test for Tutoring and Sports.
 # This will be explored further in the written report.

table(df_clean$Tutoring, df_clean$Music)
 # This code creates a contingency table regarding Tutoring and Music.
chisq.test(table(df_clean$Tutoring, df_clean$Music))
 # This code runs a chi-sq test for Tutoring and Music.
 # This will be explored further in the written report.

table(df_clean$Tutoring, df_clean$Volunteering)
 # This code creates a contingency table regarding Tutoring and 
 # Volunteering.
chisq.test(table(df_clean$Tutoring, df_clean$Volunteering))
 # This code runs a chi-sq test for Tutoring and Volunteering.
 # This will be explored further in the written report.


table(df_clean$ParentalSupport, df_clean$Extracurricular)
 # This code creates a contingency table regarding ParentalSupport and 
 # Extracurricular. 
chisq.test(table(df_clean$ParentalSupport, df_clean$Extracurricular))
 # This code runs a chi-sq test for ParentalSupport and Extracurricular.
 # This will be explored further in the written report.

table(df_clean$ParentalSupport, df_clean$Sports)
 # This code creates a contingency table regarding ParentalSupport and 
 # Sports.
chisq.test(table(df_clean$ParentalSupport, df_clean$Sports))
 # This code runs a chi-sq test for ParentalSupport and Sports.
 # This will be explored further in the written report.

table(df_clean$ParentalSupport, df_clean$Music)
 # This code creates a contingency table regarding ParentalSupport and
 # Music.
chisq.test(table(df_clean$ParentalSupport, df_clean$Music))
 # This code runs a chi-sq test for ParentalSupport and Music.
 # This will be explored further in the written report.

table(df_clean$ParentalSupport, df_clean$Volunteering)
 # This code creates a contingency table regarding ParentalSupport and
 # Volunteering.
chisq.test(table(df_clean$ParentalSupport, df_clean$Volunteering))
 # This code runs a chi-sq test for ParentalSupport and Volunteering.
 # This will be explored further in the written report.


table(df_clean$Extracurricular, df_clean$Sports)
 # This code creates a contingency table regarding Extracurricular and 
 # Sports.
chisq.test(table(df_clean$Extracurricular, df_clean$Sports))
 # This code runs a chi-sq test for Extracurricular and Sports.
 # This will be explored further in the written report.

table(df_clean$Extracurricular, df_clean$Music)
 # This code creates a contingency table regarding Extracurricular and 
 # Music.
chisq.test(table(df_clean$Extracurricular, df_clean$Music))
 # This code runs a chi-sq test for Extracurricular and Music.
 # This will be explored further in the written report.

table(df_clean$Extracurricular, df_clean$Volunteering)
 # This code creates a contingency table regarding Extracurricular and 
 # Volunteering.
chisq.test(table(df_clean$Extracurricular, df_clean$Volunteering))
 # This code runs a chi-sq test for Extracurricular and Volunteering.
 # This will be explored further in the written report.


table(df_clean$Sports, df_clean$Music)
 # This code creates a contingency table regarding Sports and Music.
chisq.test(table(df_clean$Sports, df_clean$Music))
 # This code runs a chi-sq test for Sports and Music.
 # This will be explored further in the written report.

table(df_clean$Sports, df_clean$Volunteering)
 # This code creates a contingency table regarding Sports and Volunteering.
chisq.test(table(df_clean$Sports, df_clean$Volunteering))
 # This code runs a chi-sq test for Sports and Volunteering.
 # This will be explored further in the written report.


table(df_clean$Music, df_clean$Volunteering)
 # This code creates a contingency table regarding Music and Volunteering.
chisq.test(table(df_clean$Music, df_clean$Volunteering))
 # This code runs a chi-sq test for Music and Volunteering.
 # This will be explored further in the written report.


### Categorical vs Numerical ###

summary(glm(factor(Gender) ~ StudyTimeWeekly, data = df_clean, 
            family = binomial))
summary(glm(factor(Tutoring) ~ StudyTimeWeekly, data = df_clean, 
            family = binomial))
summary(glm(factor(Extracurricular) ~ StudyTimeWeekly, data = df_clean, 
            family = binomial))
summary(glm(factor(Music) ~ StudyTimeWeekly, data = df_clean, 
            family = binomial))
summary(glm(factor(Sports) ~ StudyTimeWeekly, data = df_clean, 
            family = binomial))
summary(glm(factor(Volunteering) ~ StudyTimeWeekly, data = df_clean, 
            family = binomial))
 # This chunk of code prints a number of detailed summaries of fitted logistic
 # models comparing binary variables to StudyTimeWeekly. 
 # This will be explored further in the written report. 

kruskal.test(StudyTimeWeekly ~ Ethnicity, data = df_clean)
kruskal.test(StudyTimeWeekly ~ ParentalEducation, data = df_clean)
kruskal.test(StudyTimeWeekly ~ ParentalSupport, data = df_clean)
 # This chunk of code performs three Kruskal-Wallis test comparing the remaining 
 # variables to StudyTimeWeekly.
 # This will be explored further in the written report. 

summary(glm(factor(Gender) ~ Absences, data = df_clean, family = binomial))
summary(glm(factor(Tutoring) ~ Absences, data = df_clean, family = binomial))
summary(glm(factor(Extracurricular) ~ Absences, data = df_clean, 
            family = binomial))
summary(glm(factor(Music) ~ Absences, data = df_clean, family = binomial))
summary(glm(factor(Sports) ~ Absences, data = df_clean, family = binomial))
summary(glm(factor(Volunteering) ~ Absences, data = df_clean, 
            family = binomial))
 # This chunk of code prints a number of detailed summaries of fitted logistic
 # models comparing binary variables to Absences. 
 # This will be explored further in the written report. 

kruskal.test(Absences ~ Ethnicity, data = df_clean)
kruskal.test(Absences ~ ParentalEducation, data = df_clean)
kruskal.test(Absences ~ ParentalSupport, data = df_clean)
 # This chunk of code performs three Kruskal-Wallis test comparing the remaining 
 # variables to Absences.
 # This will be explored further in the written report. 



### Numerical vs Numerical ###

cor(df_clean$StudyTimeWeekly, df_clean$Absences, method = "spearman")
 # This code computes the Spearman rank coefficient for StudyTimeWeekly and
 # Absences.
 # This will be explored further in the written report.


################################################################################
#####                    Light Multivariate Analysis                       #####
################################################################################

modelall<- glm(GradeClass ~ Gender + Ethnicity + ParentalEducation + 
                 StudyTimeWeekly + Absences + Tutoring + ParentalSupport + 
                 Extracurricular + Sports + Music + Volunteering,
               family = binomial,
               data = df_clean)
 # This code fits a logistic regression model with GradeClass as the response
 # variable and all other variables as the exploratory variables. It labels
 # said model as 'modelall'.

summary(modelall)
 # This code prints a detailed summary of 'modelall'. This allows us to view 
 # how all the variables behave when all variables are included. 


################################################################################
#####                            Final Analysis                            #####
################################################################################

################################################################################
#####                            Model 1: GLM                              #####
################################################################################


basicmodel<- glm(GradeClass ~ Gender + Ethnicity + ParentalEducation + 
                 StudyTimeWeekly + Absences + Tutoring + ParentalSupport + 
                 Extracurricular + Sports + Music + Volunteering,
               family = binomial,
               data = df_clean)
 # This code fits a logistic regression model with GradeClass as the response
 # variable and all other variables as the exploratory variables. It labels
 # said model as 'basicmodel'.

summary(basicmodel)
 # This code prints a detailed summary of 'basicmodel'. 

best_model1 <- stepAIC(basicmodel, direction = "both")
 # This code performs a stepwise selection using AIC, it finds the model with 
 # the lowest AIC. It then labels this model as 'best_model1'.
summary(best_model1)
 # This code prints a detailed summary of 'best_model1'.

best_model2 <- stepAIC(basicmodel, direction = "both", k = log(2349))
 # This code performs a stepwise selection using BIC, it finds the model with 
 # the lowest BIC. It then labels this model as 'best_model2'.
summary(best_model2)
 # This code prints a detailed summary of 'best_model2'.


x1 <- model.matrix(GradeClass ~., df_clean)[, -1]
y1 <- df_clean$GradeClass
 # The three lines of code above, sets 'x' to be a numeric matrix of predictors
 # and 'y' to the binary variable 'GradeClass'.

set.seed(1)
train1=sample(1:nrow(x1), nrow(x1)/2)
test1 = (-train1)
y.test1 = y1[test1]
 # This code randomly splits 'x', allowing us to create a training set and a 
 # test set labelled 'train' and 'test'. The y.test, find the real entries in  
 # the 'GradeClass' column for the test data.

grid <- 10^seq(10, -2, length = 100)
ridge.mod=glmnet(x1[train1,],y1[train1],alpha=0,lambda=grid, 
                 thresh=1e-12, family= "binomial")
 # This creates a ridge regression model for the training set.

set.seed(1)
cv.out1 <- cv.glmnet(x1[train1, ], y1[train1], alpha = 0, family= "binomial")
bestlam1 <- cv.out1$lambda.min
 # These lines of code performs a cross-validation test on the ridge regression,
 # in order to find the best lambda. This is set to 'bestlam1'.

out1 <- glmnet(x1, y1, alpha = 0, family= "binomial")
ridge.coef <- predict(out1, type = "coefficients", s = bestlam1)[1:20, ]
ridge.coef 
 # This code now performs a ridge regression using the best lambda, it then 
 # extracts the coefficients and prints them.

lasso.mod=glmnet(x1[train1,],y1[train1],alpha=1,lambda=grid, family= "binomial")
 # This creates a lasso regression model for the training set.

set.seed(1)
cv.out2 <- cv.glmnet(x1[train1, ], y1[train1], alpha = 1, family= "binomial")
plot(cv.out2)
bestlam2 <- cv.out2$lambda.min
 # These lines of code performs a cross-validation test on the lasso regression,
 # in order to find the best lambda. This is set to 'bestlam2'


out2 <- glmnet(x1, y1, alpha = 1, lambda = grid, family= "binomial")
lasso.coef <- predict(out2, type = "coefficients", s = bestlam2)[1:20, ]
lasso.coef
 # This code now performs a lasso regression using the best lambda, it then 
 # extracts the coefficients and prints them.

ridge.eval <- glmnet(x1[train1,], y1[train1], alpha = 0, family= "binomial")
ridge.prob <-  predict(ridge.eval, x1[test1,], s = bestlam1,type ="response")
ridge.prob <- as.vector(ridge.prob)
ridge.pred <- rep(0, length(ridge.prob))
ridge.pred[ridge.prob > 0.5] <-1
 # This chunk of code fits a ridge regression model to the training data, then 
 # generates the predicted probability of each row in the test data to belong
 # to the 'Good' GPA class. It then classifies any probability to be greater 
 # 0.5 as to belong to the 'Good' GPA class. 
                
table(ridge.pred,y.test1)
 # This code prints the table comparing the predicted GradeClass entry to 
 # the actual GradeClass entry.
mean(ridge.pred == y.test1) 
 # This code computes the accuracy of the model. 

lasso.eval <- glmnet(x1[train1,], y1[train1], alpha = 1, family= "binomial")
lasso.prob <-  predict(lasso.eval, x1[test1,], s = bestlam2,type ="response")
lasso.prob <- as.vector(lasso.prob)
lasso.pred <- rep(0, length(lasso.prob))
lasso.pred[lasso.prob > 0.5] <-1
 # This chunk of code fits a lasso regression model to the training data, then 
 # generates the predicted probability of each row in the test data to belong
 # to the 'Good' GPA class. It then classifies any probability to be greater 
 # 0.5 as to belong to the 'Good' GPA class. 

table(lasso.pred,y.test1)
 # This code prints the table comparing the predicted GradeClass entry to 
 # the actual GradeClass entry.
mean(lasso.pred == y.test1) 
 # This code computes the accuracy of the model. 

glm.eval <- glm(GradeClass ~ StudyTimeWeekly + Absences  +
                  Tutoring + ParentalSupport + Extracurricular +
                  Sports + Music,
                family = binomial, data = df_clean, subset = train1)
 # This fits a logistic regression model using my chosen predictor variables on 
 # the training set.
glm.prob <- predict(glm.eval, df_clean[test1,], type ="response")
glm.prob <- as.vector(glm.prob)
glm.pred <- rep(0, length(glm.prob))
glm.pred[glm.prob > 0.5] <-1
 # This here predicts what GradeClass would be for the testing data based upon 
 # the trained logistic regression model.
mean(glm.pred == y.test1) 
 # This compares the predicted GradeClass entry to the actual GradeClass entry.

finallogisticmodel <- glm(GradeClass ~ StudyTimeWeekly + Absences  +
                            Tutoring + ParentalSupport + Extracurricular +
                            Sports + Music,
                          family = binomial, data = df_clean)
 # This code fits a logistic regression model with GradeClass as the response
 # variable and all our final chosen variables as the exploratory variables.
 # labels said model as 'finallogisticmodel'.

summary(finallogisticmodel)
 # This code prints a detailed summary of 'finallogisticmodel'.
 # This will be explored further in the written report. 

hnp(finallogisticmodel)
 # This code makes and plots a residual plot for the final logistic regression 
 # model.
 # This will be explored further in the written report.


################################################################################
#####                       Model 2: Deep Learning                         #####
################################################################################

n <- nrow(df_clean)
set.seed(13)
set_random_seed(13)
ntest <- trunc(n/3)
testid <- sample(1:n, ntest)
 # This chunk of code splits the data into a training set and a testing set.

x2 <- scale(model.matrix(GradeClass ~. -1, data=df_clean))
y2 <- df_clean$GradeClass
 # The three lines of code above, sets 'x' to be a numeric matrix of predictors
 # and 'y' to the binary variable 'GradeClass'.

modnn <- keras_model_sequential() %>% layer_dense(units=16, activation="relu",
                                                  input_shape=ncol(x2)) %>%
  layer_dropout(rate=0.4) %>% layer_dense(units=1, activation = "sigmoid")
 # This code sets up a model structure which describes the network.

modnn %>% compile(loss="binary_crossentropy", optimizer=optimizer_rmsprop(),
                  metrics=c("accuracy"))
 # This code add details to 'modnn' that control the fitting algorithm.

history <- modnn %>% fit(x2[-testid,], y2[-testid], epochs=300, batch_size=32,
                         validation_data=list(x2[testid,],y2[testid]) )
 # This code fits the model with the training data and two parameters epochs and
 # batch size. This is saved as a history object.
plot(history)
 # This plots the history object.


npred <- predict(modnn, x2[testid,])
 # This here predicts what GradeClass would be for the testing data based upon 
 # the trained neural network.
mean((npred > 0.5) == y2[testid])
 # This compares the predicted GradeClass entry to the actual GradeClass entry.

################################################################################
#####                              Model 3: SVM                            #####
################################################################################

set.seed(1)
x3 <- scale(model.matrix(GradeClass ~. -1, data=df_clean))
train3 <- sample(1:nrow(df_clean), nrow(df_clean)/2)
 # This chunk of code splits the data into a training set.

svmfit <- svm(factor(GradeClass) ~ ., data = df_clean[train3,], 
              kernel = "radial", gamma = 1/ncol(x3), cost = 1, scale = TRUE)
 # This fits a support vector machine model to the training data.

summary(svmfit)
 # This prints a summary of the fitted SVM.

pred.train <- predict(svmfit, df_clean[train3, ])
 # This here predicts what GradeClass would be for the testing data based upon 
 # the trained SVM.
mean(pred.train == df_clean$GradeClass[train3])
 # This compares the predicted GradeClass entry to the actual GradeClass entry.

set.seed(1)
tune.out=tune(svm, factor(GradeClass) ~., data = df_clean[train3,], 
              kernel = "radial", ranges = list(cost=c(0.1,1,10,100,1000), 
                                               gamma = c(0.01,0.1,0.5,1,2)))
 # This performs a tuning in order to find the best cost and gamma.
summary(tune.out)
 # This prints a summary of the tune out. 

pred.svm <-predict(tune.out$best.model, df_clean[-train3,])
 # This here predicts what GradeClass would be for the testing data based upon 
 # the tuned SVM.
mean(pred.svm == df_clean$GradeClass[-train3])
 # This compares the predicted GradeClass entry to the actual GradeClass entry.


################################################################################
#####                            Model 4: Random Forest                    #####
################################################################################

set.seed(1)
train4 <- sample(1:nrow(df_clean), nrow(df_clean)/2)
y.test4 <- df_clean$GradeClass[-train4]
 # This chunk of code splits the data into a training set and a testing set.

rf.df <- randomForest(factor(GradeClass) ~., data=df_clean, subset = train4, 
                      mtry = 3, importance = TRUE)
 # This fits a random forest model using the training data.

yhat.rf = predict(rf.df, df_clean[-train4,])
# This here predicts what GradeClass would be for the testing data based upon 
# the trained random forest.
mean((yhat.rf == y.test4))
 # This compares the predicted GradeClass entry to the actual GradeClass entry.

importance(rf.df)
 # This code shows two importance measures.
varImpPlot(rf.df)
 # This visualizes the two importance measures as a graph.


################################################################################
#####                             Conclusion                               #####
################################################################################

set.seed(1)
set_random_seed(1)
x <- scale(model.matrix(GradeClass ~. -1, data=df_clean))
y <- df_clean$GradeClass
train.comp <- sample(1:nrow(x), nrow(x)/2)
test.comp = (-train.comp)
y.test.comp = y[test.comp]
 # This chunk of code splits the data into a training set and a testing set.


finalglm.comp <- glm(GradeClass ~ StudyTimeWeekly + Absences  +
                            Tutoring + ParentalSupport + Extracurricular +
                            Sports + Music,
                          family = binomial,
                          data = df_clean, subset = train.comp)
pred.glm <- predict(finalglm.comp, df_clean[test.comp,], type= "response")
prob.glm <- ifelse(pred.glm > 0.5, 1, 0)
mean(prob.glm == y.test.comp)
 # This chunk of code find the testing accuracy for the final glm model.


modnn.comp <- keras_model_sequential() %>% layer_dense(units=16, #
                                                  activation="relu",
                                                  input_shape=ncol(x)) %>%
  layer_dropout(rate=0.4) %>% layer_dense(units=1, activation = "sigmoid")
modnn.comp %>% compile(loss="binary_crossentropy", 
                       optimizer=optimizer_rmsprop(), metrics=c("accuracy"))
modnn.comp %>% fit(x[train.comp,], y[train.comp], epochs=300, batch_size=32,
                         validation_data=list(x[test.comp,],y[test.comp]) )

pred.nn <- predict(modnn.comp, x[test.comp,])
prob.nn <- ifelse(pred.nn > 0.5, 1, 0)
mean(prob.nn == y.test.comp)
 # This chunk of code find the testing accuracy for the final NN model.

svmfit.opt <- svm(factor(GradeClass)~., data=df_clean[train.comp,], 
                  kernel="radial", gamma=0.01, cost=100)
pred.svm <- predict(svmfit.opt, df_clean[test.comp, ])
mean(pred.svm == y.test.comp)
 # This chunk of code find the testing accuracy for the final SVM model.

rf.opt <- randomForest(factor(GradeClass) ~., data=df_clean, 
                       subset = train.comp, mtry = 3, importance = TRUE)
pred.rf = predict(rf.opt, df_clean[test.comp,])
mean((pred.rf== y.test.comp))
 # This chunk of code find the testing accuracy for the final random forest
 # model.


table(prob.glm, y.test.comp)
table(prob.nn, y.test.comp)
table(pred.svm, y.test.comp)
table(pred.rf, y.test.comp)
 # This prints the confusion matrix for each model comparing the predicted entry
 # for GradeClass for the testing data to the true GradeClass entry.