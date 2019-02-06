##########################################
####### 1st example:
####### Simulate data with interventions
set.seed(1)
## sample size n
n <- 4000
## 5 predictor variables
p  <- 5
## simulate as independent Gaussian variables
X <- matrix(rnorm(n*p),nrow=n)
## divide data into observational (ExpInd=1) and interventional (ExpInd=2)
ExpInd <- c(rep(1,n/2),rep(2,n/2))
## for interventional data (ExpInd==2): change distribution
X[ExpInd==2,] <- sweep(X[ExpInd==2,],2, 5*rnorm(p) ,FUN="*")
## first two variables are the causal predictors of Y
beta <- c(1,1,rep(0,p-2))
## response variable Y
Y <- as.numeric(X%*%beta + rnorm(n))
## optinal: make last variable a child of Y (so last variable is non-causal for Y)
X[,p] <- 0.3*Y + rnorm(n)

####### Compute "Invariant Causal Prediction" Confidence Intervals
icp <- ICP(X,Y,ExpInd)

###### Print/plot/show summary of output
print(icp)
plot(icp)

#### compare with linear model 
cat("\n compare with linear model  \n")            
print(summary(lm(Y~X)))


##########################################
####### 2nd example:
####### Simulate a DAG where X1 -> Y, Y -> X2 and Y -> X3
####### noise interventions on second half of data on X1
####### structure of DAG (at Y -> X2) is changing under interventions
n1 <- 400
n2 <- 500
ExpInd <- c(rep(1,n1), rep(2,n2))
## index for observational (ExpInd=1) and intervention data (ExpInd=2)
X1 <- c(rnorm(n1), 2 * rnorm(n2) + 1)
Y <- 0.5 * X1 + 0.2 * rnorm(n1 + n2)
X2 <- c(1.5 * Y[1:n1]      + 0.4 * rnorm(n1),
        -0.3 * Y[(n1+1):n2] + 0.4 * rnorm(n2))
X3 <-  -0.4 * Y            + 0.2 * rnorm(n1 + n2)
X <- cbind(X1, X2, X3)

### Compute "Invariant Causal Prediction" Confidence Intervals
## use a rank-based test to detect shift in distribution of residuals
icp <- ICP(X, Y, ExpInd,test="ranks")
## use a Kolmogorov-Smirnov test to detect shift in distribution of residuals
icp <- ICP(X, Y, ExpInd,test="ks")
## can also supply test as a function
## here chosen to be equivalent to option "ks" above
icp <- ICP(X, Y, ExpInd,test=function(x,z) ks.test(x,z)$p.val)
## use a test based on normal approximations
icp <- ICP(X, Y, ExpInd, test="normal")


### Print/plot/show summary of output
print(icp)
plot(icp)

#### compare with linear model 
cat("\n compare with linear model \n")            
print(summary(lm(Y~X)))




## Not run:

##########################################
####### 3rd example:
####### College Distance data 
library(AER)
data("CollegeDistance")
CD <- CollegeDistance

##  define two experimental settings by
##  distance to closest 4-year college
ExpInd <- list()
ExpInd[[1]] <- which(CD$distance < quantile(CD$distance,0.5))
ExpInd[[2]] <- which(CD$distance >= quantile(CD$distance,0.5))

## target variable is binary (did education lead at least to BA degree?)
Y <- as.factor(CD$education>=16)
## use these predictors
X <- CD[,c("gender","ethnicity","score","fcollege","mcollege","home",
           "urban","unemp","wage","tuition","income","region")]

## searching all subsets (use selection="lasso" or selection="stability"
##     to select a subset of subsets to search)
## with selection="all" the function will take several minutes
icp <- ICP(X,Y,ExpInd,selection="all",alpha=0.1)

## Print/plot/show summary of output
print(icp)
summary(icp)
plot(icp)
## End(Not run)

