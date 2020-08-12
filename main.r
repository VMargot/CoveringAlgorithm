rm(list = ls())

args = commandArgs(trailingOnly=TRUE)
pathx = args[1]
pathy = args[2]
pathx_test = args[3]

X <- read.csv(file=pathx, header=TRUE, sep=",")
y <- read.csv(file=pathy, header=FALSE, sep=",")
y <- y$V1
X_test <- read.csv(file=pathx_test, header=TRUE, sep=",")

library(sirus)
library(nodeHarvest)

##############################
##          SIRUS           ##
##############################
sirus.m <- sirus.fit(X, y, max.depth=3, verbose=FALSE)
sirus_nbrules = length(sirus.m$rules)
int_sirus <- 0
for(i in 1:length(sirus.m$rules)){int_sirus = int_sirus + length(sirus.m$rules[[i]])}

sirus_pred <- sirus.predict(sirus.m, X_test)

sirus_rules = list()
for(i in 1:length(sirus.m$rules)){
rl = ''
dep = length(sirus.m$rules[[i]])
for(j in 1:dep){
rl = paste(rl, sirus.m$rules[[i]][[j]][1], sep='')
if(sirus.m$rules[[i]][[j]][2] == '<'){rl = paste(rl, ' in ', '-Inf;' ,sirus.m$rules[[i]][[j]][3], sep='')}
else{rl = paste(rl, ' in ', sirus.m$rules[[i]][[j]][3], ';Inf' , sep='')}
if(j < dep){rl = paste(rl, ' AND ')}
}
sirus_rules[i] = rl
}

sirus_rules <- data.frame('Rules'=matrix(unlist(sirus_rules), nrow=length(sirus_rules), byrow=T))

##############################
##       nodeharvest        ##
##############################
NH <- nodeHarvest(X, y, maxinter=3, silent=TRUE)
nh_nbrules = length(NH$nodes)
int_nh <- 0
for(i in 1:length(NH$nodes)){int_nh = int_nh + attr(NH$nodes[[i]], 'depth')}


nh_rules = list()
for(i in 1:length(NH$nodes)){
rl = ''
dep = length(NH$nodes[[i]])/3
for(j in 1:dep){
rl = paste(rl, NH$varnames[NH$nodes[[i]][j]], ' in ', NH$nodes[[i]][j+dep], ';', NH$nodes[[i]][j+2*dep], sep='')
if(j < dep){rl = paste(rl, 'AND ')}
}
nh_rules[i] = rl
}
nh_rules <- data.frame('Rules'=matrix(unlist(nh_rules), nrow=length(nh_rules), byrow=T))

nh_pred <- predict(NH, X_test)

##############################
##       Save Results       ##
##############################
path_sirus_rules <- gsub('X.csv', 'sirus_rules.csv', pathx)
path_nh_rules <- gsub('X.csv', 'nh_rules.csv', pathx)
path_sirus <- gsub('X.csv', 'sirus_pred.csv', pathx)
path_nh <- gsub('X.csv', 'nh_pred.csv', pathx)

write.csv(sirus_rules, path_sirus_rules,  row.names=FALSE)
write.csv(nh_rules, path_nh_rules,  row.names=FALSE)
write.csv(sirus_pred, path_sirus,  row.names=FALSE)
write.csv(nh_pred, path_nh,  row.names=FALSE)

