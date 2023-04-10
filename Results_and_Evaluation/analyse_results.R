setwd("~/PATH/OF/RESULTS")
library(readr)
results <- data.frame(read_table2("results_Ours.txt", 
                         col_names = FALSE))
if(results[2,2] == "biased:" | results[2,2] == "signal:" | results[2,2] == "r_statistic:" | results[2,2] == "vanilla:"){
  results[,1] <- paste(results[,1],results[,2],sep="_")
  results <- results[,-2]
}
colnames(results) <- c("Experiment","Test accuracy")

# Remove breaks and order dataset
results <- results[is.na(results[,5]),1:2]
results <- as.data.frame(results[order(results$Experiment), ])
results[,2] <- as.numeric(results[,2])

# Count how many seeds we set
n_seeds <- max(which(results[,1]==results[1,1]))
n_exp <- nrow(results) / n_seeds
if (n_exp%%1!=0){quit(save="ask")}


# Calculate mean&var
mean_var <- data.frame(Experiment=NA,Mean=NA,Standard_Deviation=NA)
for(i in 1:n_exp){
  mean_var[i,1] <- results[((i-1)*n_seeds+1),1]
  mean_var[i,2] <- mean(results[((i-1)*n_seeds+1):(i*n_seeds),2])*100
  mean_var[i,3] <- sd(results[((i-1)*n_seeds+1):(i*n_seeds),2])*100
}
View(mean_var)


#Manually saving every baseline's results variable by their own name

#--------------------CIFAR10C:-------------------
# p-value = 0.1088
t.test(vanilla[vanilla[,1]=="Vanilla_Cifar10C_0.2pct_vanilla:",2],Ours[Ours[,1]=="CIFAR10C_0.2pct_signal:",2],alternative = "two.sided", paired=TRUE)
# p-value = 0.002686
t.test(vanilla[vanilla[,1]=="Vanilla_Cifar10C_0.2pct_vanilla:",2],LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.2-Severity4_signal:",2],alternative = "two.sided",paired=TRUE)

# p-value = 0.176
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.1-Severity4_signal:",2],Ours[Ours[,1]=="CIFAR10C_0.1pct_signal:",2],alternative = "two.sided",paired=TRUE)
# p-value = 4.587e-05
t.test(vanilla[vanilla[,1]=="Vanilla_Cifar10C_0.1pct_vanilla:",2],Ours[Ours[,1]=="CIFAR10C_0.1pct_signal:",2],alternative = "two.sided",paired=TRUE)


# p-value = 0.8521
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.05-Severity4_signal:",2],Ours[Ours[,1]=="CIFAR10C_0.05pct_signal:",2],alternative = "two.sided",paired=TRUE)
# p-value = 3.777e-06
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.05-Severity4_signal:",2],vanilla[vanilla[,1]=="Vanilla_Cifar10C_0.05pct_vanilla:",2],alternative = "two.sided",paired=TRUE)


# p-value = 0.006337
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.02-Severity4_signal:",2],Ours[Ours[,1]=="CIFAR10C_0.02pct_signal:",2],alternative = "two.sided",paired=TRUE)
# p-value = 5.336e-05
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.02-Severity4_signal:",2],Dfa[Dfa[,1]=="cifar10c_2_ours_signal:",2],alternative = "two.sided",paired=TRUE)

# p-value = 0.0003114
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.01-Severity4_signal:",2],Ours[Ours[,1]=="CIFAR10C_0.01pct_signal:",2],alternative = "two.sided",paired=TRUE)

# p-value = 0.1665
t.test(LfF[LfF[,1]=="CorruptedCIFAR10-Type0-Skewed0.005-Severity4_signal:",2],Dfa[Dfa[,1]=="cifar10c_0.5_ours_signal:",2],alternative = "two.sided",paired=TRUE)
# p-value = 0.0005805
t.test(Ours[Ours[,1]=="CIFAR10C_0.005pct_signal:",2],Dfa[Dfa[,1]=="cifar10c_0.5_ours_signal:",2],alternative = "two.sided",paired=TRUE)



#--------------------CMNIST:-------------------
# p-value = 1.913e-05
t.test(vanilla[vanilla[,1]=="CMNIST_0.2pct_vanilla:",2],Dfa[Dfa[,1]=="cmnist_20_ours_signal:",2],alternative = "two.sided",paired=TRUE)

# p-value = 0.001072
t.test(vanilla[vanilla[,1]=="CMNIST_0.1pct_vanilla:",2],Dfa[Dfa[,1]=="cmnist_10_ours_signal:",2],alternative = "two.sided", paired=TRUE)

# p-value = 0.2923
t.test(vanilla[vanilla[,1]=="CMNIST_0.05pct_vanilla:",2],Ours[Ours[,1]=="CMNIST_0.05pct_signal:",2],alternative = "two.sided", paired=TRUE)
# p-value = 0.4492
t.test(Dfa[Dfa[,1]=="cmnist_5_ours_signal:",2],Ours[Ours[,1]=="CMNIST_0.05pct_signal:",2],alternative = "two.sided", paired=TRUE)
# p-value = 0.0002139
t.test(LfF[LfF[,1]=="ColoredMNIST-Skewed0.05-Severity4_signal:",2],Ours[Ours[,1]=="CMNIST_0.05pct_signal:",2],alternative = "two.sided", paired=TRUE)

# p-value = 0.08379
t.test(Dfa[Dfa[,1]=="cmnist_2_ours_signal:",2],Ours[Ours[,1]=="CMNIST_0.02pct_signal:",2],alternative = "two.sided", paired=TRUE)
# p-value = 3.395e-05
t.test(LfF[LfF[,1]=="ColoredMNIST-Skewed0.02-Severity4_signal:",2],Ours[Ours[,1]=="CMNIST_0.02pct_signal:",2],alternative = "two.sided", paired=TRUE)

# p-value = 0.01083
t.test(Dfa[Dfa[,1]=="cmnist_1_ours_signal:",2],Ours[Ours[,1]=="CMNIST_0.01pct_signal:",2],alternative = "two.sided", paired=TRUE)

# p-value = 0.001608
t.test(Dfa[Dfa[,1]=="cmnist_0.5_ours_signal:",2],Ours[Ours[,1]=="CMNIST_0.005pct_signal:",2],alternative = "two.sided", paired=TRUE)
