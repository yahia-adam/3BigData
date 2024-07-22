# Note prés,



1) quand on a terminé l'implem de model on savais donc on a lance plein de model avec des config au hazard:
   
   1) diff lr, epoch, nbr couche, nbr neurone.
   
   2) le loads de dataset prends enormement de temps donc on stocke sur format binare et le loads le binaire.
   
   3) dataset insufisant donc pour respectecter un peu pres 10 image * nbr de neurones, on a resize les image et on a fait 32*32
   
   4) on avais un loss et une accuracy parfait, mais quand on testais, avec des image ca marchais pas.
   
   5) on a alos ajouté les test loss et le train et on a observé que le model generalise tress
   
   6) on a observe que le model ne generalise pas bien sur les donné de teste alors tres dificille a debug.
      
      1) pram (ouverfittin/ underfiting)
      
      2) lr ()
      
      3) dataset (faire matrice de confusion, test avec keras)
   
   7)  
