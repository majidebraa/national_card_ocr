"""Classical card detection and perspective rectification (no ML, CPU only)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idcard_ocr.errors import CardNotFoundError

CARD_ASPECT = 85.6 / 54.0  # ISO/IEC 7810 ID-1
ASPECT_TOLERANCE = 0.55  # relative; phone photos are often taken at an angle (foreshortening)
MIN_AREA_FRACTION = 0.10  # first pass; small cards get a second pass (SMALL_AREA_FRACTION)
SMALL_AREA_FRACTION = 0.03
GOOD_SCORE = 0.62
MIN_ACCEPT = 0.18
NEAR_BEST = 0.9


@dataclass(frozen=True, slots=True)
class Detection:
    quad: np.ndarray  # 4x2 float32 in source-image pixels, ordered tl, tr, br, bl
    confidence: float
    method: str


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _edge_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    tl, tr, br, bl = quad
    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    return top, bottom, left, right


def _aspect(quad: np.ndarray) -> float:
    top, bottom, left, right = _edge_lengths(quad)
    w = (top + bottom) / 2.0
    h = (left + right) / 2.0
    if min(w, h) < 1e-3:
        return 0.0
    return max(w, h) / min(w, h)


def _edge_support(quad: np.ndarray, grad: np.ndarray) -> float:
    """Fraction of points along the quad's sides that lie on a strong image edge (0..1)."""
    h, w = grad.shape
    hits = 0
    total = 0
    for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        n = max(8, int(np.linalg.norm(b - a) / 4))
        for t in np.linspace(0.05, 0.95, n):
            x, y = a + (b - a) * t
            xi, yi = round(float(x)), round(float(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue
            total += 1
            win = grad[max(0, yi - 3) : yi + 4, max(0, xi - 3) : xi + 4]
            hits += int(win.size > 0 and float(win.max()) > 0.35)
    return hits / total if total else 0.0


def _content_score(quad: np.ndarray, image_bgr: np.ndarray) -> float:
    """How much the quad's interior looks like a printed card rather than a background patch.

    Deliberately generic (it must hold for every card design): a card carries some colour, unlike
    grayscale crops of a portrait or a neutral table, and some detail, unlike flat patches of
    wall or print. Returns 0..1.
    """
    src = order_quad(quad)
    dst = np.array([[0, 0], [159, 0], [159, 100], [0, 100]], dtype=np.float32)
    warped = cv2.warpPerspective(image_bgr, cv2.getPerspectiveTransform(src, dst), (160, 101))
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = float(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0).mean())
    detail = float(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).std())
    colour_term = min(1.0, max(0.0, (chroma - 3.0) / 5.0))
    detail_term = min(1.0, max(0.0, (detail - 15.0) / 15.0))
    return colour_term * detail_term


def _score_candidate(
    quad: np.ndarray,
    contour_area: float,
    frame_area: float,
    grad: np.ndarray | None = None,
    min_area: float = MIN_AREA_FRACTION,
    image: np.ndarray | None = None,
) -> float:
    aspect = _aspect(quad)
    if aspect <= 0:
        return 0.0
    aspect_err = abs(aspect - CARD_ASPECT) / CARD_ASPECT
    if aspect_err > ASPECT_TOLERANCE:
        return 0.0
    quad_area = float(cv2.contourArea(quad.reshape(-1, 1, 2)))
    if quad_area < min_area * frame_area:
        return 0.0
    rectangularity = min(1.0, contour_area / max(quad_area, 1.0))
    size_term = min(1.0, quad_area / (0.5 * frame_area)) ** 0.5
    aspect_term = 1.0 - aspect_err / ASPECT_TOLERANCE
    base = 0.45 * rectangularity + 0.30 * aspect_term + 0.25 * size_term
    if grad is None:
        return float(base)
    # Quads that do not follow real image edges (background merged in) are penalised.
    score = 0.6 * base + 0.4 * _edge_support(quad, grad)
    if image is not None:
        score *= 0.25 + 0.75 * _content_score(quad, image)
    return float(score)


def _candidates_from_mask(
    mask: np.ndarray,
    frame_area: float,
    grad: np.ndarray | None = None,
    min_area: float = MIN_AREA_FRACTION,
    image: np.ndarray | None = None,
) -> list[tuple[float, np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    results: list[tuple[float, np.ndarray]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area < min_area * frame_area:
            break
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        quad: np.ndarray | None = None
        for eps in (0.02, 0.035, 0.05, 0.07):
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                quad = order_quad(approx.reshape(4, 2))
                break
        if quad is None:
            rect = cv2.minAreaRect(hull)
            quad = order_quad(cv2.boxPoints(rect))
        score = _score_candidate(
            quad, float(cv2.contourArea(hull)), frame_area, grad, min_area, image
        )
        if score > 0:
            results.append((score, quad))
    return results


def _foreground_masks(small: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    masks: list[tuple[str, np.ndarray]] = []

    # 1) colour distance from the border (the card is usually surrounded by background)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate(
        [
            lab[:6].reshape(-1, 3),
            lab[-6:].reshape(-1, 3),
            lab[:, :6].reshape(-1, 3),
            lab[:, -6:].reshape(-1, 3),
        ]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist_u8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    _, m1 = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, kernel, iterations=2)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, kernel, iterations=1)
    masks.append(("border-distance", m1))

    # 2) Canny edges, closed into a solid shape
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    filled = np.zeros_like(edges)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    masks.append(("canny", filled))

    # 3) Otsu on brightness
    _, m3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m3 = cv2.morphologyEx(m3, cv2.MORPH_CLOSE, kernel, iterations=2)
    masks.append(("otsu", m3))

    # 4) saturation: pastel printed cards stand out against neutral tables and walls
    sat = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)
    _, m4 = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m4 = cv2.morphologyEx(m4, cv2.MORPH_CLOSE, kernel, iterations=3)
    m4 = cv2.morphologyEx(m4, cv2.MORPH_OPEN, kernel, iterations=2)
    masks.append(("saturation", m4))

    # 5) pastel pink/blue: the printed card's own colours, which skin and neutral backgrounds lack
    light, a, b = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0
    raw = ((light > 140) & ((b < -3) | ((a > 6) & (b < 6)))).astype(np.uint8) * 255
    m5 = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=3)
    m5 = cv2.morphologyEx(m5, cv2.MORPH_OPEN, kernel, iterations=2)
    masks.append(("pastel", m5))
    masks.append(("otsu-inv", cv2.bitwise_not(m3)))
    return masks


def detect_card(image_bgr: np.ndarray, *, max_side: int = 960) -> Detection:
    """Find the card quadrilateral. Raises :class:`CardNotFoundError` if none is plausible."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = (
        cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image_bgr
    )
    sh, sw = small.shape[:2]
    frame_area = float(sh * sw)

    if float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std()) < 8.0:
        raise CardNotFoundError("The image is blank or has almost no contrast.")

    gray_small = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    grad = cv2.magnitude(
        cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3),
    )
    grad = np.clip(grad / max(float(np.percentile(grad, 99)), 1.0), 0.0, 1.0)

    best: tuple[float, np.ndarray, str] | None = None
    masks = _foreground_masks(small)
    for min_area in (MIN_AREA_FRACTION, SMALL_AREA_FRACTION):
        found: list[tuple[float, np.ndarray, str]] = []
        for name, mask in masks:
            for score, quad in _candidates_from_mask(mask, frame_area, grad, min_area, small):
                found.append((score, quad, name))
        if found:
            top = max(f[0] for f in found)
            # Internal printed regions (a header band, a photo box) can look as card-like as the
            # card itself, so among near-equal candidates the outermost one wins.
            near = [f for f in found if f[0] >= NEAR_BEST * top]
            pick = max(near, key=lambda f: float(cv2.contourArea(f[1].reshape(-1, 1, 2))))
            if best is None or pick[0] > best[0]:
                best = pick
        if best is not None and best[0] >= GOOD_SCORE:
            break

    # The photo may be a flat scan with no surrounding background at all.
    full_aspect = max(sh, sw) / max(1, min(sh, sw))
    has_content = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).std()) > 12.0
    if (
        has_content
        and abs(full_aspect - CARD_ASPECT) / CARD_ASPECT < 0.12
        and (best is None or best[0] < 0.55)
    ):
        quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
        best = (0.5, quad, "full-frame")

    if best is None or best[0] < MIN_ACCEPT:
        raise CardNotFoundError(
            "No card-shaped region found. Place the card on a plain background, fully in frame."
        )
    score, quad, method = best
    return Detection(quad=(quad / scale).astype(np.float32), confidence=float(score), method=method)


def rectify(image_bgr: np.ndarray, quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp the quad to a fronto-parallel ``size`` (width, height) image, long edge horizontal."""
    top, bottom, left, right = _edge_lengths(quad)
    src = quad.astype(np.float32)
    if (top + bottom) < (left + right):
        # The card was photographed in portrait orientation: rotate the corner order by 90 deg.
        src = np.roll(src, -1, axis=0)
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        image_bgr, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def quad_width_px(quad: np.ndarray) -> float:
    """Length (source pixels) of the card's long edge, used for the resolution check."""
    top, bottom, left, right = _edge_lengths(quad)
    return max((top + bottom) / 2.0, (left + right) / 2.0)
