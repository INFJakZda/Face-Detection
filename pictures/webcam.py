import pygame.camera
import pygame.image
import time

pygame.camera.init()
cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
cam.start()
time.sleep(0.4)
img = cam.get_image()

pygame.image.save(img, "img.jpg")
pygame.camera.quit()
