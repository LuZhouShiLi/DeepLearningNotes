library(forestplot)
library(patchwork)
library(ggplot2)
result <- read.csv('C:/Users/Admini/Desktop/senlintulag01.csv')
# C:\Users\Admini\Desktop



forestplot(result[,c(1,2,3,7)], 
           mean=result[,4],   
           lower=result[,5],  
           upper=result[,6], 
           zero=1,            
           boxsize=0.4,      
           graph.pos= 4,
           hrzl_lines=list("1" = gpar(lty=1,lwd=2),
                           "2" = gpar(lty=1),
                           "12" = gpar(lty=2),
                           "22" = gpar(lty=2),
                           "32" = gpar(lty=2),
                           "42" = gpar(lty=2),
                           "52" = gpar(lty=2),
                           "62" = gpar(lwd=2,lty=1)),
           graphwidth = unit(.25,"npc"),
           xlab="",
           xticks=c(0.8,0.9,1,1.1,1.2),
           is.summary=c(T,T,F,F,
                        T,F,F,F,
                        T,F,F,
                        T,F,F,
                        T,F,F,F,
                        T,F,F,
                        T,F,F,
                        T,F,F,F,
                        T,F,F,
                        T,F,F,
                        T,F,F,F,
                        T,F,F,
                        T,F,F,
                        T,F,F,F,
                        T,F,F,
                        T,F,F,
                        T,F,F,F,
                        T,F,F
                        
           ),#T=粗体
           txt_gp=fpTxtGp(label=gpar(cex=0.4),
                          ticks=gpar(cex=0.5), 
                          xlab=gpar(cex=1), 
                          title=gpar(cex=2)),
           lwd.zero=1,
           lwd.ci=1.2,
           lwd.xaxis=1, 
           lty.ci=1,
           ci.vertices =T,
           ci.vertices.height=0.1, 
           clip=c(0,3),
           ineheight=unit(8, 'mm'), 
           line.margin=unit(8, 'mm'),
           colgap=unit(6, 'mm'),
           col=fpColors(zero =  "#808080",
                        box = '#048410', 
                        lines ="black"
           ),
)