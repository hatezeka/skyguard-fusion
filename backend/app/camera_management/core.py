"""SkyGuard camera profile registry. Stdlib-only, no hardware discovery claims."""
from __future__ import annotations
import json, math, os, tempfile
from urllib.parse import urlsplit
from pathlib import Path
from datetime import datetime, timezone

class CameraError(ValueError): pass

def number(v, name, low=None, high=None, strict_low=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise CameraError(f'{name}: finite number required')
    if low is not None and (v <= low if strict_low else v < low): raise CameraError(f'{name}: below allowed minimum')
    if high is not None and v > high: raise CameraError(f'{name}: above allowed maximum')
    return float(v)

def validate_source_location(source_type, location):
    if location is None:
        return

    if source_type == "usb":
        if type(location) is not int or not 0 <= location <= 4:
            raise CameraError("USB source_location must be index 0-4")
        return

    if source_type == "rtsp":
        if not isinstance(location, str):
            raise CameraError("RTSP source_location must be URL")

        try:
            parsed = urlsplit(location)
            valid = (
                parsed.scheme.lower() == "rtsp"
                and bool(parsed.hostname)
                and not parsed.fragment
                and not any(ch.isspace() for ch in location)
            )
        except ValueError:
            valid = False

        if not valid:
            raise CameraError("Invalid RTSP source_location")

        if parsed.username is not None or parsed.password is not None:
            raise CameraError(
                "RTSP credentials are not allowed in camera profiles"
            )
        return

    raise CameraError("source_location unsupported for source type")


def validate(profile):
    if not isinstance(profile, dict): raise CameraError('profile must be object')
    if not isinstance(profile.get('camera_id'), str) or not profile['camera_id'].strip(): raise CameraError('camera_id required')
    if profile.get('source_type') not in ('usb','rtsp','virtual'): raise CameraError('invalid source_type')
    validate_source_location(profile['source_type'], profile.get('source_location'))
    s=profile.get('stream',{})
    if not isinstance(s,dict): raise CameraError('stream must be object')
    for k in ('width_px','height_px'):
        v=s.get(k)
        if v is not None and (isinstance(v,bool) or not isinstance(v,int) or v<=0): raise CameraError(k+' must be positive integer')
    if s.get('fps') is not None: number(s['fps'],'fps',0,strict_low=True)
    geo=profile.get('geolocation',{})
    if not isinstance(geo,dict): raise CameraError('geolocation must be object')
    if geo.get('source') not in ('unavailable','device','external_gnss','manual','simulation'): raise CameraError('invalid location source')
    lat,lon=geo.get('latitude'),geo.get('longitude')
    if (lat is None)!=(lon is None): raise CameraError('latitude and longitude required together')
    if lat is not None:
        number(lat,'latitude',-90,90); number(lon,'longitude',-180,180)
        if geo['source']=='unavailable': raise CameraError('location source required')
    if geo.get('altitude_m') is not None: number(geo['altitude_m'],'altitude_m')
    if geo.get('horizontal_accuracy_m') is not None: number(geo['horizontal_accuracy_m'],'horizontal_accuracy_m',0)
    cal=profile.get('calibration',{})
    if not isinstance(cal,dict): raise CameraError('calibration must be object')
    if cal.get('status') not in ('UNCALIBRATED','CALIBRATED','INVALIDATED'): raise CameraError('invalid calibration status')
    intrinsic=cal.get('intrinsics')
    if intrinsic is not None:
        if not isinstance(intrinsic,dict): raise CameraError('intrinsics must be object')
        for k in ('fx','fy','cx','cy'):
            if k not in intrinsic: raise CameraError('missing '+k)
            number(intrinsic[k],k,0 if k in ('fx','fy') else None,strict_low=k in ('fx','fy'))
        dist=intrinsic.get('distortion')
        if not isinstance(dist,list) or len(dist)!=5: raise CameraError('5 distortion coefficients required')
        for i,v in enumerate(dist): number(v,f'distortion[{i}]')
    if cal['status']=='CALIBRATED':
        if intrinsic is None or cal.get('resolution')!=[s.get('width_px'),s.get('height_px')]:
            raise CameraError('calibration requires intrinsics matching current resolution')
        if not cal.get('verified_at'): raise CameraError('verified_at required')
    return profile

class CameraRegistry:
    def __init__(self, path):
        self.path=Path(path)
        if self.path.exists():
            data=json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data,dict): raise CameraError('invalid registry file')
            for p in data.values(): validate(p)
            self._items=data
        else: self._items={}

    def _save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        fd,tmp=tempfile.mkstemp(prefix='.cameras-',suffix='.json',dir=self.path.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                json.dump(self._items,f,indent=2,ensure_ascii=False)
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp,self.path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def register(self,camera_id,source_type,stream=None,source_location=None):
        if camera_id in self._items: raise CameraError('camera already registered')
        p={'camera_id':camera_id,'source_type':source_type,'source_location':source_location,'stream':stream or {},
           'geolocation':{'latitude':None,'longitude':None,'altitude_m':None,'horizontal_accuracy_m':None,'source':'unavailable','captured_at':None},
           'calibration':{'status':'UNCALIBRATED','intrinsics':None,'resolution':None,'verified_at':None},
           'orientation':{'yaw_deg':None,'pitch_deg':None,'roll_deg':None,'source':'unavailable'}}
        validate(p);self._items[camera_id]=p;self._save();return self.get(camera_id)

    def get(self,camera_id):
        if camera_id not in self._items: raise CameraError('unknown camera')
        return json.loads(json.dumps(self._items[camera_id]))

    def list(self): return [self.get(k) for k in sorted(self._items)]

    def update_stream(self,camera_id,stream):
        p=self.get(camera_id)
        if not isinstance(stream,dict): raise CameraError('stream must be object')
        previous=p['stream'];p['stream']=stream
        if (previous.get('width_px'),previous.get('height_px')) != (stream.get('width_px'),stream.get('height_px')) and p['calibration']['status']=='CALIBRATED':
            p['calibration']['status']='INVALIDATED'
        return self._commit(camera_id,p)

    def set_location(self,camera_id,latitude,longitude,source,altitude_m=None,horizontal_accuracy_m=None):
        p=self.get(camera_id)
        p['geolocation']={'latitude':latitude,'longitude':longitude,'altitude_m':altitude_m,
          'horizontal_accuracy_m':horizontal_accuracy_m,'source':source,
          'captured_at':datetime.now(timezone.utc).isoformat()}
        return self._commit(camera_id,p)

    def set_calibration(self,camera_id,intrinsics,verified_at=None):
        p=self.get(camera_id);s=p['stream']
        p['calibration']={'status':'CALIBRATED','intrinsics':intrinsics,
          'resolution':[s.get('width_px'),s.get('height_px')],
          'verified_at':verified_at or datetime.now(timezone.utc).isoformat()}
        return self._commit(camera_id,p)

    def import_usb_calibration(self, camera_id, candidate, optical_setup):
        import hashlib

        profile = self.get(camera_id)
        if profile["source_type"] != "usb" or not optical_setup.strip():
            raise CameraError("USB camera and explicit optical setup required")
        try:
            width, height = candidate["image_size_px"]
            matrix = candidate["camera_matrix"]
            distortion = candidate["distortion_coefficients"]
            rms = candidate["global_reprojection_rms_px"]
            if len(distortion) == 1 and isinstance(distortion[0], list):
                distortion = distortion[0]
            fx, _, cx = matrix[0]
            _, fy, cy = matrix[1]
            if matrix[0][1] != 0 or matrix[1][0] != 0 or matrix[2] != [0, 0, 1]:
                raise ValueError("Invalid camera matrix")
            if type(width) is not int or type(height) is not int:
                raise ValueError("Invalid image size")
            number(cx, "cx", 0, width - 1)
            number(cy, "cy", 0, height - 1)
            number(rms, "reprojection RMS", 0)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise CameraError("Invalid USB calibration profile") from exc
        number(fx, "fx", 0, strict_low=True)
        number(fy, "fy", 0, strict_low=True)
        if not isinstance(distortion, list) or len(distortion) != 5:
            raise CameraError("Five distortion coefficients required")
        for coefficient in distortion:
            number(coefficient, "distortion")
        warning = candidate.get("quality_warning")
        if warning is not None and type(warning) is not bool:
            raise CameraError("Invalid quality warning")
        if (width, height) not in ((640, 480), (1280, 720), (1920, 1080)):
            raise CameraError("Unsupported USB calibration resolution")
        calibration = {
            "status": "CALIBRATED",
            "validation_status": "CALIBRATED_UNVALIDATED",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "resolution": [width, height],
            "capture_backend": "DSHOW",
            "capture_codec": "MJPG",
            "optical_setup": optical_setup.strip(),
            "reprojection_rms_px": rms,
            "quality_warning": candidate.get("quality_warning"),
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "distortion": distortion},
        }
        payload = {key: value for key, value in calibration.items() if key != "verified_at"}
        profile_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        calibration["profile_id"] = profile_id
        versions = profile.setdefault("calibration_profiles", {})
        profile["calibration"] = versions.setdefault(profile_id, calibration)
        profile["stream"].update(width_px=width, height_px=height)
        return self._commit(camera_id, profile)

    def set_orientation(self,camera_id,yaw_deg,pitch_deg,roll_deg,source,north_reference=None,evidence_ref=None):
        p=self.get(camera_id)
        for name,v in [('yaw_deg',yaw_deg),('pitch_deg',pitch_deg),('roll_deg',roll_deg)]: number(v,name)
        if source not in ('sensor','manual','calibration','simulation'): raise CameraError('invalid orientation source')
        if north_reference is not None or evidence_ref is not None:
            if (source != 'calibration' or north_reference not in ('true_north', 'magnetic_north')
                or not isinstance(evidence_ref, str) or not evidence_ref.strip()):
                raise CameraError('Referenced orientation requires calibrated source and evidence')
            if not 0 <= yaw_deg < 360 or not -90 <= pitch_deg <= 90 or not -180 <= roll_deg <= 180:
                raise CameraError('Orientation angles outside supported bounds')
        p['orientation']={'yaw_deg':yaw_deg,'pitch_deg':pitch_deg,'roll_deg':roll_deg,'source':source}
        if north_reference is not None:
            p['orientation'].update(north_reference=north_reference,evidence_ref=evidence_ref.strip())
        return self._commit(camera_id,p)

    def readiness(self,camera_id):
        p=self.get(camera_id);g=p['geolocation'];o=p['orientation'];c=p['calibration']
        return {'camera_id':camera_id,'detection_ready':bool(p['stream'].get('width_px') and p['stream'].get('height_px')),
          'ray_geometry_ready':c['status']=='CALIBRATED' and all(o.get(k) is not None for k in ('yaw_deg','pitch_deg','roll_deg')),
          'camera_geolocation_ready':g['latitude'] is not None and g['longitude'] is not None and g['altitude_m'] is not None,
          'drone_depth_observable':False,'note':'camera GPS and calibration alone cannot measure drone range'}

    def _commit(self,camera_id,p):
        validate(p);self._items[camera_id]=p;self._save();return self.get(camera_id)
