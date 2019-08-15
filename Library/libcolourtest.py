import libcvzero

image_path= "test2.jpg"

cvzero_learn = libcvzero.Learn()

cvzero_learn.image = image_path

cvzero_learn.get_dominant_colour()



