
rm(list = ls())
graphics.off()

library(tidyverse)
library(readxl)

# Import and preprocess HIV incidence data
hiv_rate <- read_excel("hiv_rate.xlsx")
hiv_rate <- filter(hiv_rate, Age == "ALLAGE")
hiv_rate <- filter(hiv_rate, Sex == "BOTHSEX")
hiv_rate <- hiv_rate[,c(7,12:15)]
hiv_rate <- gather(hiv_rate, key = year, value = value, 2:5)
hiv_rate$year <- as.numeric(hiv_rate$year)
hiv_rate$value <- as.numeric(hiv_rate$value)
names(hiv_rate) <- c('country', 'year', 'hiv_rate')

# Import and preprocess HIV aid data
hiv_aid <- read.csv("hiv_aid.csv", skip = 4, header = FALSE)
hiv_aid <- hiv_aid[,c(2,4:7)]
names(hiv_aid) <- c('country', '2014', '2015', '2016','2017')
for (i in 2:dim(hiv_aid)[2]) {
  for (j in 1:dim(hiv_aid)[1]) {
    if (hiv_aid[j,i] == '..'){
      hiv_aid[j,i] <- gsub('..', NA, hiv_aid[j,i])
    }
  }
}
no_miss <- rep(NA, dim(hiv_aid)[1])
for (i in 1:dim(hiv_aid)[1]){
  no_miss[i] <- sum(is.na(hiv_aid[i,]))
}
ind_miss <- which(no_miss > 2)
hiv_aid <- hiv_aid[-ind_miss,]
hiv_aid <- gather(hiv_aid, key = year, value = value, 2:5)
hiv_aid$year <- as.numeric(hiv_aid$year)
hiv_aid$value <- as.numeric(hiv_aid$value)
names(hiv_aid) <- c('country', 'year', 'hiv_aid')

# Import and preprocess macroeconomic data
gdp_per_cap_PPP <- read.csv("gdp_per_cap_PPP.csv", skip = 4)
gdp_growth <- read.csv("gdp_growth.csv", skip = 4)
fdi <- read.csv("fdi.csv", skip = 4)
inflation <- read.csv("inflation.csv", skip = 4)
unemployment <- read.csv("unemployment.csv", skip = 4)
population <- read.csv("population.csv", skip = 4)
fertility <- read.csv("fertility.csv", skip = 4)
maternal_mort <- read.csv("maternal_mort.csv", skip = 4)
infant_mort <- read.csv("infant_mort.csv", skip = 4)
tuberculosis <- read.csv("tuberculosis.csv", skip = 4)
life_exp <- read.csv("life_exp.csv", skip = 4)
school_enr <- read.csv("school_enr.csv", skip = 4)
electricity <- read.csv("electricity.csv", skip = 4)
undernourishment <- read.csv("undernourishment.csv", skip = 4)
covariates <- data.frame(rbind(gdp_per_cap_PPP, gdp_growth, inflation, fdi,  
                         unemployment, population, fertility, tuberculosis,
                         maternal_mort, infant_mort, life_exp, school_enr,
                         electricity, undernourishment))
covariates <- covariates[,c(1,2,3,59:62)]
covariates <- gather(covariates, key = year, value = value, 4:7)
covariates <- spread(covariates, key = Indicator.Name, value = value)
names(covariates) <- c('country', 'id', 'year', 'electricity', 'fertility', 'fdi', 
                       'gdp_growth', 'gdp_per_cap_PPP','tuberculosis','inflation',
                       'life_exp', 'maternal_mort','infant_mort','population', 
                       'undernourishment', 'school_enr', 'unemployment')
covariates$year <- rep(2014:2017,length(unique(covariates$country)))
covariates$fdi <- covariates$fdi/1000000
covariates$gdp_per_cap_PPP <- covariates$gdp_per_cap_PPP/1000
covariates$population <- covariates$population/1000000
covariates$school_enr <- covariates$school_enr/100

# Align country names and merge (aid + covariates)
ctry_names1 <- unique(hiv_aid$country)
ctry_names2 <- unique(covariates$country)
ctry_names1[which(! ctry_names1 %in% ctry_names2)] # 33
ctry_names2
hiv_aid$country[which(hiv_aid$country == "China (People's Republic of)")] <- "China"
hiv_aid$country[which(hiv_aid$country == "Congo")] <- "Congo, Rep."
hiv_aid$country[which(hiv_aid$country == "Democratic Republic of the Congo")] <- "Congo, Dem. Rep."
hiv_aid$country[which(hiv_aid$country == "Democratic People's Republic of Korea")] <- "Korea, Dem. People's Rep."
hiv_aid$country[which(hiv_aid$country == "Egypt")] <- "Egypt, Arab Rep."
hiv_aid$country[which(hiv_aid$country == "Gambia")] <- "Gambia, The"
hiv_aid$country[which(hiv_aid$country == "Iran")] <- "Iran, Islamic Rep."
hiv_aid$country[which(hiv_aid$country == "Kyrgyzstan")] <- "Kyrgyz Republic"
hiv_aid$country[which(hiv_aid$country == "Lao People's Democratic Republic")] <- "Lao PDR"
hiv_aid$country[which(hiv_aid$country == "Viet Nam")] <- "Vietnam"
hiv_aid$country[which(hiv_aid$country == "Venezuela")] <- "Venezuela, RB"
hiv_aid$country[which(hiv_aid$country == "West Bank and Gaza Strip")] <- "West Bank and Gaza"
hiv_aid$country[which(hiv_aid$country == "Yemen")] <- "Yemen, Rep."
ctry_names1 <- unique(hiv_aid$country)
ctry_names1[which(! ctry_names1 %in% ctry_names2)] # 20 now
aid_covariates <- merge(hiv_aid, covariates, by=c('country', 'year'))

# Align country names and merge (hiv_rate + aid_covariates)
ctry_names1 <- unique(hiv_rate$country)
ctry_names2 <- unique(aid_covariates$country)
ctry_names1[which(! ctry_names1 %in% ctry_names2)] # 40
ctry_names2
hiv_rate$country[which(hiv_rate$country == "Bolivia (Plurinational State of)")] <- "Bolivia"
hiv_rate$country[which(hiv_rate$country == "Congo")] <- "Congo, Rep."
hiv_rate$country[which(hiv_rate$country == "Côte d'Ivoire")] <- "Cote d'Ivoire"
hiv_rate$country[which(hiv_rate$country == "Democratic Republic of the Congo")] <- "Congo, Dem. Rep."
hiv_rate$country[which(hiv_rate$country == "Egypt")] <- "Egypt, Arab Rep."
hiv_rate$country[which(hiv_rate$country == "Gambia")] <- "Gambia, The"
hiv_rate$country[which(hiv_rate$country == "Iran (Islamic Republic of)")] <- "Iran, Islamic Rep."
hiv_rate$country[which(hiv_rate$country == "Kyrgyzstan")] <- "Kyrgyz Republic"
hiv_rate$country[which(hiv_rate$country == "Lao People's Democratic Republic")] <- "Lao PDR"
hiv_rate$country[which(hiv_rate$country == "Republic of Moldova")] <- "Moldova"
hiv_rate$country[which(hiv_rate$country == "United Republic of Tanzania")] <- "Tanzania"
hiv_rate$country[which(hiv_rate$country == "Viet Nam")] <- "Vietnam"
hiv_rate$country[which(hiv_rate$country == "Venezuela (Bolivarian Republic of)")] <- "Venezuela, RB"
hiv_rate$country[which(hiv_rate$country == "Yemen")] <- "Yemen, Rep."
ctry_names1 <- unique(hiv_rate$country)
ctry_names1[which(! ctry_names1 %in% ctry_names2)] # 26 now
HIV_final <- merge(hiv_rate, aid_covariates, by=c('country', 'year'))
HIV_final <- HIV_final[,c(1,5,2:4,6:19)]

# Store final version
write.csv(HIV_final, file = "zHIV_data.csv", row.names = FALSE)





