################################################################################
usage <- function() {
	stop("USAGE: Rscript Th_experiments.R <Th_model_file.bnet> <sync/async> <nrows> <ncols>", call.=FALSE)
}
################################################################################

args = commandArgs(trailingOnly=TRUE)
if (length(args)!=4) { usage() }
model_file <- args[1]
if (args[2]=="sync") {
	policy <- "synchronous"
} else if (args[2]=="async") {
	policy <- "asynchronous"
} else { usage() }
nrows <- as.numeric(args[3])
ncols <- as.numeric(args[4])
runs <- nrows * ncols

################################################################################
# determine successor state (synchronous or asynchronous)
async_succ <- function(model, state, policy) {
	tmpsucc <- stateTransition(model, state, type="synchronous")
	if (policy=="synchronous") {
		return(tmpsucc)
	}
	diffs <- tmpsucc[state != tmpsucc]
	succ <- state
	if (length(diffs)>0) {
		pos <- sample(1:length(diffs), 1)
		succ[[names(diffs)[pos]]] <- diffs[pos]
	}
	return(succ)
}

################################################################################
library(BoolNet)
# load the model file and simulate
PBNth<-loadNetwork(model_file,bodySeparator = ",",lowercaseGenes = FALSE,symbolic = FALSE)
countTh0<-0
countTh1<-0
countTh2<-0
Th<-vector(mode="numeric",length=runs);
for (i in 1:runs) {
	# initial state all to 0 but IFNg
	curr <- generateState(PBNth, specs=c("IFNg"=1))
	# determine successor state
	succ <- async_succ(PBNth, curr, policy)

	while (!identical(curr,succ)){ # continue until stable state
		curr <- succ
		succ <- async_succ(PBNth, curr, policy)
	}
	if (succ[['GATA3']]==1) { #Th2
		countTh2<-countTh2+1
		Th[i]<-2
	} else {
		if (succ[['Tbet']]==1) {#Th1
			countTh1<-countTh1+1
			Th[i]<-1
		} else { #Th0
			countTh0<-countTh0+1
			Th[i]<-0
		}
	}
}


ThTab<-matrix(Th,nrow=nrows,ncol=ncols)
if (countTh1 ==0 & countTh2 ==0) {
	colors <- c('0' = "green")
} else if (countTh0 ==0 & countTh2 ==0) {
	colors <- c('1' = "red") 
} else if (countTh0 ==0 & countTh2 ==0) {
	colors <- c('1' = "red")
} else if (countTh0 ==0 & countTh2 !=0 & countTh1 !=0) {
	colors <- c('1' = "red",'2'="blue")
} else if (countTh0 !=0 & countTh2 !=0 & countTh1 !=0) {
	colors <- c('0'="green",'1' = "red",'2'="blue")
}	else if (countTh0 !=0 & countTh1 !=0 & countTh2 ==0) {
	colors <- c('0'="green",'1' = "red")
}

#-------------------------------------------------------------------------------
base_file <- paste(tools::file_path_sans_ext(model_file), ".", args[2], sep="")

#-------------------------------------------------------------------------------
cat("Total runs: ", runs,
		"\n# runs Th0: ", countTh0, "(", format(round(100*countTh0/runs, 2), nsmall=1), "%)",
		"\n# runs Th1: ", countTh1, "(", format(round(100*countTh1/runs, 2), nsmall=1), "%)",
		"\n# runs Th2: ", countTh2, "(", format(round(100*countTh2/runs, 2), nsmall=1), "%)",
		"\n", file=paste(base_file,".txt", sep=""))

#-------------------------------------------------------------------------------
pdf(paste(base_file,".pdf",sep=""),
		width=ncols*2, height=nrows*2, pointsize=10)
par(mar=c(0.0,0.0,0.0,0.0))
image(1:ncol(ThTab), 1:nrow(ThTab), as.matrix(t(ThTab)), col=colors,
			xaxt="n", yaxt="n", bty="n", xlab="", ylab="",asp=1)
dev.off()

