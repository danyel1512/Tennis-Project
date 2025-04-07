def convert_pixels_to_meters(pixel_dist,ref_height_meters,ref_height_pixels):
    return (pixel_dist * ref_height_meters) / ref_height_pixels

def convert_meters_to_pixels(meters,ref_height_meters,ref_height_pixels):
    return (meters * ref_height_pixels) / ref_height_meters
