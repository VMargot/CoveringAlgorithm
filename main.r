rm(list = ls())

args = commandArgs(trailingOnly=TRUE)
pathx = args[1]
pathy = args[2]
pathx_test = args[3]

X <- read.csv(file=pathx, header=TRUE, sep=",")
y <- read.csv(file=pathy, header=FALSE, sep=",")
y <- y$V1
X_test <- read.csv(file=pathx_test, header=TRUE, sep=",")

##############################
##          SIRUS           ##
##############################

library(sirus)
sirus.m <- sirus.fit(X, y, verbose=FALSE)
sirus_nbrules = length(sirus.m$rules)
int_sirus <- 0
for(i in 1:length(sirus.m$rules)){int_sirus = int_sirus + length(sirus.m$rules[[i]])}

sirus_pred <- sirus.predict(sirus.m, X_test)

##############################
##       nodeharvest        ##
##############################

library(nodeHarvest)
NH <- nodeHarvest(X, y, silent=TRUE)
nh_nbrules = length(NH$nodes)
int_nh <- 0
for(i in 1:length(NH$nodes)){int_nh = int_nh + attr(NH$nodes[[i]], 'depth')}

nh_pred <- predict(NH, X_test)

##############################
##       Save Results       ##
##############################

path_sirus <- gsub('X.csv', 'sirus_pred.csv', pathx)
path_nh <- gsub('X.csv', 'nh_pred.csv', pathx)
path_int <- gsub('X.csv', 'int.csv', pathx)

write.csv(sirus_pred, path_sirus,  row.names=FALSE)
write.csv(nh_pred, path_nh,  row.names=FALSE)
t <- data.frame('Sirus' = c(sirus_nbrules, int_sirus), 'NH' = c(nh_nbrules, int_nh), row.names=c('nb_rules', 'int'))
write.csv(t, path_int, row.names=TRUE)
