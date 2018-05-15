from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#Sample inputs.
# point = Point(0.5, 0.5)
# polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
def point_in_polygon(point, poly_points):
	'''This function will take the indecies of a polygon and determine if a 
	given point lies within that polygon.
	input: 	point as tuple
			poly_points as list of tuples
	returns:T/F'''
	poly = Polygon([poly_points])
	return polygon.contains(point)