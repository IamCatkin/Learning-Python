library(GOSemSim)
data = read.csv('D:\\Python\\project\\bindingdb\\20181227\\pro\\uni.csv',row.names=1)
genelist = list(as.character(data[,2]))
onts = c('BP', 'MF', 'CC')
measures = c('Resnik', 'Wang', 'Lin', 'Rel', 'Jiang')
for ( i in onts) {
  print(paste(i,',start!'))
  d = godata('org.Hs.eg.db', ont=i)  
  for (j in measures) {
    print(paste(j,',start!'))
    gsimmat = mgeneSim(unlist(genelist), semData=d, measure=j,drop=NULL)
    filename = paste('D:\\Python\\project\\bindingdb\\20181227\\pro\\gsimmat_',j,'_',i,'.csv',sep='')
    write.csv(gsimmat,filename)
    print(paste(j,'_',i,',done!',sep=''))
  }
}