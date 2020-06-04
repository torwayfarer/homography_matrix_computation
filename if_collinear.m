% function to determine if there are collinear points among the 4 points
function f=if_collinear(points_locations)
    
    f=false;
    indexes=[1 2 3; 1 2 4; 2 3 4; 1 3 4];
    
    for i = 1 : size(indexes)
        index = indexes(i,:);
        points=points_locations(index, :);

        x1=points(1, 1);
        y1=points(1, 2);
        x2=points(2, 1);
        y2=points(2, 2);
        x3=points(3, 1);
        y3=points(3, 2);
        
        if ((y3 - y2) * (x2 - x1) == (y2 - y1) * (x3 - x2))
            f=true;
            break
        end
        
    end
    
end