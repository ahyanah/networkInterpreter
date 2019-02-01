# ChangeLog

changes in autoAnn 29th Jan
[x]- OTHER -> OTHERS
[x]- GREY -> SILVER
[x]- CAR -> SEDAN
[x]- added BLACK to colour detector
[x]- removed colour tag for MOTORCYCLE
[x]- add annotation, images, categories and licenses
[ ] sort out file hierarchy

[x] remove stopsigns, TRAIN, TRAFFIC LIGHT, bicycle

Changed 
ALL SILVER, in categories to SILVER
	TRUCK, SILVER, -> TRUCK, SILVER
	BUS, SILVER, -> BUS, SILVER
	MVP, SILVER, -> MVP, SILVER
	SEDAN, SILVER, -> "SEDAN, SILVER"

# Installation:
After downloading his script and env.ini 
You can choose to install Mask-RcNN in any location. 
clone Mask-RCNN from https://github.com/matterport/Mask_RCNN
-- requires manual pip installation of Shapely.whl
-- requires coco API via >>>> pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI