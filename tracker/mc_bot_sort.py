import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from copy import deepcopy

#from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.cls = -1
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(cls, score)

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

        self.rois = []
        self.path_history = []
        self.in_roi = False
        self.last_crossing = 0
        self.was_in_roi = False
        self.in_crossing_id = None
        self.was_out_roi = False
        self.out_crossing_id = None

        self.messages = {}


    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score
                    found = True

                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            self.cls_hist.append([cls, score])
            self.cls = cls
            
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

        self.update_cls(new_track.cls, new_track.score)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        
        new_tlwh = new_track.tlwh
        bec = self.tlwh_to_bec(new_tlwh)
        
        points_in_roi = {}
        for id, roi in self.rois.items():
            if self.point_in_roi(bec, roi):
                points_in_roi[id] = True
            else:
                points_in_roi[id] = False

        self.path_history.append({"point": bec, 
                                  "in_roi": points_in_roi})

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)

        if self.rois:

            if len(self.path_history) < 3:
                return {}
            
            current_point = self.path_history[-1]
            previous_point = self.path_history[-2]
            previous_previous_point = self.path_history[-3]

            # if self.track_id <= 1.1:
            #     print(current_point)
            #     print(previous_point)

            event = None

            if any((previous_point["in_roi"]["1_0"], previous_previous_point["in_roi"]["1_0"])) and current_point["in_roi"]["1_1"]:
                event = 1
                print(event, self.track_id)
            elif any((previous_point["in_roi"]["1_1"], previous_previous_point["in_roi"]["1_1"])) and current_point["in_roi"]["1_0"]:
                event = 0
                print(event, self.track_id)

            if event is not None:
                message = {
                    "roi": id,
                    "frontier": 0,
                    "class": int(self.cls),
                    "action": event,
                    "track_id": self.track_id}

                msg_key = str(self.track_id)
                if msg_key not in self.messages:
                    self.messages[msg_key] = deepcopy(message)
            
                    return message

            # for id, roi in self.rois.items():

            #     if current_point["in_roi"][id] and (not previous_point["in_roi"][id]):
            #         self.in_crossing_id = self.check_crossing([previous_point["point"], current_point["point"]], 
            #                                                   roi, 
            #                                                   leaving=False)
            #         if self.in_crossing_id == 2:
            #             message = {
            #             "roi": id,
            #             "frontier": 0,
            #             "class": int(self.cls),
            #             "action": 1,
            #             "track_id": self.track_id}
            #             msg_key = str(id)+"_"+str(0)
            #             if msg_key not in self.messages:
            #                 self.messages[str(id)+"_"+str(0)] = deepcopy(message)
            #         else:
            #             message = {
            #                 "roi": id,
            #                 "frontier": self.in_crossing_id,
            #                 "class": int(self.cls),
            #                 "action": 0,
            #                 "track_id": self.track_id}
            #             msg_key = str(id)+"_"+str(self.in_crossing_id)
            #             if msg_key not in self.messages:
            #                 self.messages[str(id)+"_"+str(self.in_crossing_id)] = deepcopy(message)
            #         #print("messages in track: ", self.messages)
            #         return message
                
            #     if (not current_point["in_roi"][id]) and previous_point["in_roi"][id]:
            #         self.out_crossing_id = self.check_crossing([previous_point["point"], current_point["point"]], 
            #                                                    roi, 
            #                                                    leaving=True)
            #         if self.out_crossing_id == 2:
            #             message = {
            #             "roi": id,
            #             "frontier": 0,
            #             "class": int(self.cls),
            #             "action": 0,
            #             "track_id": self.track_id}
            #             msg_key = str(id)+"_"+str(0)
            #             if msg_key not in self.messages:
            #                 self.messages[str(id)+"_"+str(0)] = deepcopy(message)
            #         else:
            #             message = {
            #                 "roi": id,
            #                 "frontier": self.out_crossing_id,
            #                 "class": int(self.cls),
            #                 "action": 1,
            #                 "track_id": self.track_id}
            #             msg_key = str(id)+"_"+str(self.out_crossing_id)
            #             if msg_key not in self.messages:
            #                 self.messages[str(id)+"_"+str(self.out_crossing_id)] = deepcopy(message)
                    
            #         return message

        return {}
    
    def get_messages(self):
        return self.messages

    @staticmethod
    def check_crossing(path, polygon, leaving = True):

        num_vertices = len(polygon)
        #TODO DESMARTELAR
        lines = [(polygon[i-3], polygon[i - 4]) for i in range(num_vertices)]
        path_len = len(path)
        cross_id = None

        for j in range(path_len-1):
            for i in range(len(lines)):
                roi_line = lines[i]
                if not leaving:
                    path_line = (path[j], path[j+1])
                else:
                    path_line = (path[-j-1], path[-j-2])
                #check if path_line crosses roi_line
                if STrack.do_intersect(path_line, roi_line):
                    cross_id = i
                    break
            if cross_id is not None:
                break

        return cross_id

    
    @staticmethod
    def orientation(p, q, r):
        """Return the orientation of the triplet (p, q, r).
        0 -> p, q and r are collinear
        1 -> Clockwise
        2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    @staticmethod
    def on_segment(p, q, r):
        """Check if point r lies on segment pq."""
        if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
            return True
        return False

    @staticmethod
    def do_intersect(seg1, seg2):
        p1, q1 = seg1
        p2, q2 = seg2

        # Find the four orientations needed for the general and special cases
        o1 = STrack.orientation(p1, q1, p2)
        o2 = STrack.orientation(p1, q1, q2)
        o3 = STrack.orientation(p2, q2, p1)
        o4 = STrack.orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special cases
        # p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if o1 == 0 and STrack.on_segment(p1, q1, p2):
            return True

        # p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if o2 == 0 and STrack.on_segment(p1, q1, q2):
            return True

        # p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if o3 == 0 and STrack.on_segment(p2, q2, p1):
            return True

        # p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if o4 == 0 and STrack.on_segment(p2, q2, q1):
            return True

        # Doesn't fall in any of the above cases
        return False


    @staticmethod
    def point_in_roi(point, polygon):
        num_vertices = len(polygon)
        x, y = point[0], point[1]
        inside = False
    
        # Store the first point in the polygon and initialize the second point
        p1 = polygon[0]
    
        # Loop through each edge in the polygon
        for i in range(1, num_vertices + 1):
            # Get the next point in the polygon
            p2 = polygon[i % num_vertices]
    
            # Check if the point is above the minimum y coordinate of the edge
            if y > min(p1[1], p2[1]):
                # Check if the point is below the maximum y coordinate of the edge
                if y <= max(p1[1], p2[1]):
                    # Check if the point is to the left of the maximum x coordinate of the edge
                    if x <= max(p1[0], p2[0]):
                        # Calculate the x-intersection of the line connecting the point to the edge
                        x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
    
                        # Check if the point is on the same line as the edge or to the left of the x-intersection
                        if p1[0] == p2[0] or x <= x_intersection:
                            # Flip the inside flag
                            inside = not inside
    
            # Store the current point as the first point for the next iteration
            p1 = p2
    
        # Return the value of the inside flag
        return inside

    @staticmethod
    def tlwh_to_bec(tlwh):
        """Convert bounding box to bottom edge center (x,y) format."""
        return np.array([tlwh[0] + tlwh[2]/2, tlwh[1] + tlwh[3]])


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, args, frame_rate=5, rois = []):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

        #roi
        self.rois = rois

    def update(self, output_results, img, event_on_lost_track = False):

        events = []

        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            classes = output_results[:, 5]
            features = output_results[:, 6:]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            features = output_results[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            features_keep = features[remain_inds]
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                              (tlbr, s, c, f) in zip(dets, scores_keep, classes_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                              (tlbr, s, c) in zip(dets, scores_keep, classes_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                #create track
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
            
            #TODO this is not very elegant
            if not track.rois:
                track.rois = self.rois
            
        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        if not self.args.mot20:
            ious_dists = matching.fuse_score(ious_dists, detections)

        if self.args.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                message = track.update(detections[idet], self.frame_id)
                if message and not event_on_lost_track:
                    events.append(message)
                    print(message)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                                 (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                message = track.update(det, self.frame_id)
                if message and not event_on_lost_track:
                    events.append(message)
                    print(message)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            message = unconfirmed[itracked].update(detections[idet], self.frame_id)
            if message and not event_on_lost_track:
                events.append(message)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            #Removed Tracks
            track = unconfirmed[it]
            # message = track.track_left()
            track.mark_removed()
            # if event_on_lost_track:
            #     messages = track.get_messages()
                #print("MESSAGES IN TRACKER (UNCONFIRMED): ", messages)
                # for msg_id, msg in messages.items():
                #     events.append(msg)
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                #Removed Tracks
                if event_on_lost_track and track.state != TrackState.Removed:
                    messages = track.get_messages()
                    #print("MESSAGES IN TRACKER (UPDATE STATE): ", messages)
                    for msg_id, msg in messages.items():
                        msg["state"] = track.state
                        events.append(msg)
                track.mark_removed()
                removed_stracks.append(track)


        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]


        return output_stracks, events
    
    def get_tracks_np(self):
        tracks = []
        for track in self.tracked_stracks:
            tracks.append([track.tlwh[0],
                            track.tlwh[1],
                            track.tlwh[0]+track.tlwh[2],
                            track.tlwh[1]+track.tlwh[3], 
                            track.score, 
                            max(track.cls_hist, key=lambda x: x[1])[0], 
                            track.track_id])
        return np.array(tracks)
    
def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
