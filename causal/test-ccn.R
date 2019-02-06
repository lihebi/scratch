## Testing CompareCausalNetworks package


## Simulate data with connectivity matrix A with assumptions 1) No
## hidden variables 2) Precise location of interventions is known

require(backShift)
require(CompareCausalNetworks)
require(pcalg)
require(InvariantCausalPrediction)

## Use this to close the X11 window
## graphics.off()



## sample size n
n <- 10000

## p=5 predictor variables and connectivity matrix A
p <- 5
labels <- c("1", "2", "3", "4", "5")

generateA <- function (p) {
    A <- diag(p)*0
    A[1,2] <- 0.8
    A[2,3] <- -0.8
    A[3,4] <- 0.8
    A[3,5] <- 0.8
    A[4,5] <- 0.3
    ## can add/remove feedback by using/not using
    ## A[5,2] <- 0.8 
    return (A)
}

A = generateA(p)

## simulate data
myseed <- 1
## divide data in 10 different environments
G <- 10

## simulate choose explicity intervention targets
simResult <- backShift::simulateInterventions(n, p, A, G,
                                              intervMultiplier = 3,
                                              noiseMult = 1,
                                              nonGauss = TRUE,
                                              hiddenVars = FALSE,
                                              knownInterventions = TRUE,
                                              fracVarInt = 0.2,
                                              simulateObs = TRUE,
                                              seed = myseed)

X <- simResult$X
environment <- simResult$environment
interventions <- simResult$interventions

## number of unique environments
G <- length(unique(environment))


## apply all  methods given in vector 'methods'
methods <- c("ICP", "hiddenICP", "gies")

## select whether you want to run stability selection
stability <- FALSE

## X <- A
## Option 1): use this estimator as a point estimate
Ahat_ICP <- getParents(X, environment, interventions = interventions,
                   method="ICP", alpha=0.1, pointConf = TRUE)
Ahat_hiddenICP <- getParents(X, environment, interventions = interventions,
                   method="hiddenICP", alpha=0.1, pointConf = TRUE)
Ahat_GIES <- getParents(X, environment, interventions = interventions,
                        method="gies", alpha=0.1, pointConf = TRUE)
Ahat_backShift <- getParents(X, environment, interventions = interventions,
                        method="backShift", alpha=0.1, pointConf = TRUE)



##############################
## Printing numerical results
cat("\n true graph is  ------  \n" )
print(A)
cat("\n result for method", method,"  ------  \n" )
## print and plot estimate (point estimate thresholded if
## numerical estimates are returned)
print(Ahat_ICP)
print(Ahat_hiddenICP)
print(Ahat_GIES)

##############################
## plotting

## arrange graphical output into a rectangular grid
sq <- ceiling(sqrt(length(methods)+1))
par(mfrow=c(ceiling((length(methods)+1)/sq),sq))
## plot and print true graph
plotGraphEdgeAttr(A, plotStabSelec = FALSE, labels = labels, 
                  thres.point = 0, main = "TRUE GRAPH")
plotGraphEdgeAttr(Ahat_ICP, plotStabSelec = FALSE,
                  labels = labels, thres.point = 0.05,
                  main=paste("POINT ESTIMATE FOR ICP\n"))
plotGraphEdgeAttr(Ahat_hiddenICP, plotStabSelec = FALSE,
                  labels = labels, thres.point = 0.05,
                  main=paste("POINT ESTIMATE FOR hiddenICP\n"))
plotGraphEdgeAttr(Ahat_backShift, plotStabSelec = FALSE,
                  labels = labels, thres.point = 0.05,
                  main=paste("POINT ESTIMATE FOR GIES\n"))


