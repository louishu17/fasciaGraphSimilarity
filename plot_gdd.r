library(ggplot2)

dat = read.csv(header=T, 'test.out.gdd')

ggplot(data = dat, aes(x = Count, y=Nodes)) + 
  geom_point() +
  labs(x="Graphlet Degree",size=12) +
  labs(y="# of Nodes with degree",size=12) +
  scale_y_log10() +
  scale_x_log10() +
  theme_bw() + 
  theme(axis.title = element_text(size=24))

ggsave("out_gdd.png", width=6, height=4.3)
