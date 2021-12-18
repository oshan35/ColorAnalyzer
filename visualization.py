import matplotlib.pyplot as plt
import cv2

"""
file contains visualization methods this is not a part of main implementation. In case of validations 
we can use this methods.
"""

def generate_pie_chart(counts, rgb_colors, hex_colors, image):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1)
    # resized_image = cv2.resize(image, (1200, 600))
    original_image = plt.imshow(image)
    ax.set_title('Original Image')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    fig.tight_layout(pad=5.0)

    ax = fig.add_subplot(1, 2, 2)
    pie_chart = plt.pie(counts.values(), labels=rgb_colors, colors=hex_colors)
    ax.set_title('color chart')

    plt.show()

    plt.savefig('figures/result.png')
