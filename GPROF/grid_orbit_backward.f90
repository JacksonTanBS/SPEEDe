subroutine grid_orbit_backward(lons_orbit, lats_orbit, var_orbit, &
                               ntrack, nswath, &
                               lons_grid, lats_grid, nlon, nlat, &
                               max_dist, var_grid)

!    Jackson Tan (jackson.tan@nasa.gov)
!    Last updated: 2023 10 25

!    This subroutine performs the backward gridding of an orbit file
!    into a cylindrical equidistant grid. For each grid box, backward 
!    gridding seeks the nearest footprint (within a maximum distance 
!    threshold) and uses its value as is (i.e., nearest neighbor 
!    gridding). Hence, each footprint may be mapped to multiple grid
!    boxes. This is sometimes called "nearest source to destination"
!    gridding.

!    The inputs are the 2D longitude, latitude, and variable arrays 
!    from the orbit (dimensions: track, swath), as well as the 1D 
!    longitude and latitude arrays of the cylindrical grid, and the 
!    maximum distance threshold.

!    The gridding loops over each footprint, compares the distance to 
!    the grid boxes, and assigns the value to that grid box if the 
!    distance within the maximum threshold and is nearer than previous 
!    footprints. To speed up the process, in the loop for each footprint 
!    along the track, the subroutine will record the grid box with the 
!    minimum distance from the previous footprint and search the 
!    neighborhood of this grid box. This is set by the flag 
!    "search_neighborhood".

    implicit none

    integer*4 :: nlon, nlat, ntrack, nswath

    real*4    :: lons_orbit(ntrack, nswath), &
                 lats_orbit(ntrack, nswath), &
                 var_orbit(ntrack, nswath), &
                 lons_grid(nlon), &
                 lats_grid(nlat), &
                 dist_grid(nlon, nlat), &
                 var_grid(nlon, nlat)

    integer*4 :: t, s, i0, j0, i1, j1, ni, nj, min_i1, min_j1, fv_i, ioffset

    ! neighboring grid boxes to search for
    integer*4 :: delta_i(51) = (/ (i0, i0 = -25, 25, 1) /)
    integer*4 :: delta_j(51) = (/ (j0, j0 = -25, 25, 1) /)

    real*4    :: dist, max_dist, min_dist, fv_f

    real*4    :: dist_rect

    logical   :: search_neighborhood

    !f2py intent(in) :: lons_orbit, lats_orbit, var_orbit
    !f2py intent(in) :: lons_grid, lats_grid, max_dist
    !f2py intent(out) :: var_grid

!    Define the parameters.

    fv_f = -9999.9    ! floating-point missing value
    fv_i = -9999      ! integer missing value
    search_neighborhood = .true.

!    Initialize the grid arrays.

    dist_grid(:, :) = max_dist
    var_grid(:, :) = fv_f

!    Loop over each swath (cross-track).

    do s = 1, nswath

!    Initialize the "previous" grid box indices.

        i0 = fv_i
        j0 = fv_i

!    Loop over each track (along-track).

        do t = 1, ntrack

!    Skip if coordinates are flagged as missing.

            if (lons_orbit(t, s) .eq. fv_f .or. &
                lats_orbit(t, s) .eq. fv_f) then
                cycle
            end if

!    If there is a valid "previous" grid box, search the neighborhood 
!    of the previous grid box.

            if (i0 .ne. fv_i .and. j0 .ne. fv_i .and. &
                search_neighborhood) then

                min_dist = max_dist
                min_i1 = fv_i
                min_j1 = fv_i

                do nj = 1, size(delta_j)

!    Calculate the neighboring latitude index, wrapping around the poles 
!    if necessary.

                    j1 = j0 + delta_j(nj)
                    if (j1 .lt. 1) then
                        j1 = 1 - j1
                        ioffset = 180
                    else if (j1 .gt. nlat) then
                        j1 = nlat * 2 - j1 + 1
                        ioffset = 180
                    else
                        ioffset = 0
                    end if

                    do ni = 1, size(delta_i)

!    Calculate the neighboring longitude index.

                        i1 = modulo(i0 + delta_i(ni) + ioffset - 1, nlon) + 1

!    Calculate the distance from the footprint to the grid box.

                        dist = dist_rect(lons_orbit(t, s), lons_grid(i1), &
                                         lats_orbit(t, s), lats_grid(j1))

!    If distance is lower than currently recorded grid value, overwrite 
!    the existing grid values.

                        if (dist .lt. dist_grid(i1, j1) .and. &
                            var_orbit(t, s) .ne. fv_f) then
                            dist_grid(i1, j1) = dist
                            var_grid(i1, j1) = var_orbit(t, s)
                        end if

!    If distance is lower than any for this footprint, record it for 
!    the subsequent footprint.

                        if (dist .lt. min_dist) then
                            min_dist = dist
                            min_i1 = i1
                            min_j1 = j1
                        end if

                    end do
                end do

!    Set the "previous" indices for the next footprint.

                i0 = min_i1
                j0 = min_j1

            end if

!    If there is not a valid "previous" grid box or if the neighbor search
!    fails to pick up a minimum valud, loop through grid boxes to 
!    find the closest grid.

            if (i0 .eq. fv_i .or. j0 .eq. fv_i .or. &
                .not. search_neighborhood) then

                min_dist = max_dist
                min_i1 = fv_i
                min_j1 = fv_i

                do j1 = 1, nlat
                    do i1 = 1, nlon

!    Calculate the distance from the footprint to the grid box.

                        dist = dist_rect(lons_orbit(t, s), lons_grid(i1), &
                                         lats_orbit(t, s), lats_grid(j1))

!    If distance is lower than currently recorded grid value, overwrite 
!    the existing grid values.

                        if (dist .lt. dist_grid(i1, j1) .and. &
                            var_orbit(t, s) .ne. fv_f) then
                            dist_grid(i1, j1) = dist
                            var_grid(i1, j1) = var_orbit(t, s)
                        end if

!    If distance is lower than any for this footprint, record it for 
!    the subsequent footprint.

                        if (dist .lt. min_dist) then
                            min_dist = dist
                            min_i1 = i1
                            min_j1 = j1
                        end if

                    end do
                end do

!    Set the "previous" indices for the next footprint.

                i0 = min_i1
                j0 = min_j1

            end if

        end do
    end do

end subroutine

real function haversine(lon0, lon1, lat0, lat1)

    ! Last updated: 2022 09 08

    implicit none
    real*4, intent(in) :: lon0, lon1, lat0, lat1
    real*4 :: rlon0, rlon1, rlat0, rlat1, R, deg2rad

    R = 6371    ! mean radius of Earth
    deg2rad = 0.0174533

    rlon0 = lon0 * deg2rad
    rlon1 = lon1 * deg2rad
    rlat0 = lat0 * deg2rad
    rlat1 = lat1 * deg2rad

    haversine = 2 * R * asin(sqrt(sin((rlat1 - rlat0) / 2) ** 2 + &
                             cos(rlat0) * cos(rlat1) * &
                             sin((rlon1 - rlon0) / 2) ** 2))

end function

real function dist_rect(lon0, lon1, lat0, lat1)

    ! Cylindrical approximation for physical distance; alternative to 
    ! Haversine but faster; valid for small distances. Returns the 
    ! square of the distance.

    implicit none
    real*4, intent(in) :: lon0, lon1, lat0, lat1
    real*4 :: R, deg2rad, dist_lat, dist_lon

    R = 6371    ! mean radius of Earth
    deg2rad = 0.0174533

    dist_lat = (lat1 - lat0) * deg2rad
    dist_lon = (lon1 - lon0) * deg2rad * cos((lat0 + lat1) / 2 * deg2rad)

    dist_rect = R * sqrt(dist_lon ** 2 + dist_lat ** 2)

end function