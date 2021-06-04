
# Deep learning for detecting three lung sicknesss from X-Ray iamges



* Pneumocystis pneumonia (PCP) is a serious infection that causes inflammation and fluid buildup in your lungs. It's brought on by a fungus called Pneumocystis jirovecii that spreads through the air. This fungus is very common. Most people's immune systems have fought it off by the time they're 3 or 4 years old.

<p align="center">
  <img src="Assets/Pneu.PNG" >
</p>

ref: https://www.google.com/search?q=Pneumocystis&oq=Pneumocystis&aqs=chrome.0.69i59j46i433j0i433l4j0l4.1176j0j7&sourceid=chrome&ie=UTF-8

* COVID-19 is caused by a coronavirus called SARS-CoV-2. Older adults and people who have severe underlying medical conditions like heart or lung disease or diabetes seem to be at higher risk for developing more serious complications from COVID-19 illness.

<p align="center">
  <img hight=100 src="Assets/covid.png" >
</p>

* Streptococcus is a genus of gram-positive coccus or spherical bacteria that belongs to the family Streptococcaceae, within the order Lactobacillales, in the phylum Firmicutes. Cell division in streptococci occurs along a single axis, so as they grow, they tend to form pairs or chains that may appear bent or twisted.





# CNN model to detect COVID from patients X-Ray images

Deep learning for covid 19 detections. Two sets of data including X-Ray images of Normal and covid patients were gathered.
The CNN developed and was applied for detecting COVID cases from normal. In the second part of this project, I will use the CNN m for detecting 20 different categories.

* Below dataset images after applying augmentation are seen.

<p align="center">
  <img src="Assets/plot_01_assets_2.png" >
</p>


* The CNN model different metrics are seen after 10 epoches.
<p align="center">
  <img src="Figures/plot_01_1.png" >
</p>




oversampling
class weigting means multplying loss function
