args = commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
  stop("USAGE: Rscript Th_experiments.R <Th_model_file.bnet> <runs>", call.=FALSE)
}
model_file <- args[1]
runs <- as.numeric(args[2])

library(BoolNet)
# load the model file and simulate
PBNth<-loadNetwork(model_file,bodySeparator = ",",lowercaseGenes = FALSE,symbolic = FALSE)
countTh0<-0
countTh1<-0
countTh2<-0
Th<-vector(mode="numeric",length=runs);
for (i in 1:runs) {
	# initial state all to 0 but IFNg
	init <- generateState(PBNth, specs=c("IFNg"=1))
	# determine sucessor state (synchronous but probabilistic)
	succ<-stateTransition(PBNth,init,type="probabilistic")

	while (!identical(init,succ)){ # continue until stable state
		init<-succ
		succ<-stateTransition(PBNth,init,type="probabilistic")
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


ThTab<-matrix(Th,nrow=100,ncol=10)
if (countTh1 ==0 & countTh2 ==0){
	cols <- c('0' = "green")} else {
		if (countTh0 ==0 & countTh2 ==0) {
			cols <- c('1' = "red")} else {
				if (countTh0 ==0 & countTh2 ==0) {
					cols <- c('1' = "red")} else {
						if (countTh0 ==0 & countTh2 !=0 & countTh1 !=0){
							cols <- c('1' = "red",'2'="blue")} else {
								if (countTh0 !=0 & countTh2 !=0 & countTh1 !=0){
									cols <- c('0'="green",'1' = "red",'2'="blue")}	else {
										if (countTh0 !=0 & countTh1 !=0 & countTh2 ==0){
											cols <- c('0'="green",'1' = "red")
										}
								}
						}
				}
		}
}

#-------------------------------------------------------------------------------
base_file <- tools::file_path_sans_ext(model_file)

#-------------------------------------------------------------------------------
cat("Total runs: ", runs,
		"\n# runs Th0: ", countTh0, "(", format(round(100*countTh0/runs, 2), nsmall=1), "%)",
		"\n# runs Th1: ", countTh1, "(", format(round(100*countTh1/runs, 2), nsmall=1), "%)",
		"\n# runs Th2: ", countTh2, "(", format(round(100*countTh2/runs, 2), nsmall=1), "%)",
		"\n", file=paste(base_file,".txt", sep=""))

#-------------------------------------------------------------------------------
savepdf <- function(file, width=16, height=10)
{
  pdf(file, width=width/1.91, height=height/12.0, pointsize=10)
  par(mar=c(0.0,0.0,0.0,0.0))
}
savepdf(file=paste(base_file,".pdf",sep=""))
image(1:nrow(ThTab), 1:ncol(ThTab), as.matrix(ThTab), col=cols,xaxt="n", yaxt="n", bty="n", xlab="", ylab="",asp=1)
dev.off()
