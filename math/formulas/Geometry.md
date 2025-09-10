Geometry is a **must-have toolbox** in competitive programming (CP).
Here’s a **complete Geometry Formula Handbook** in **Markdown format**, structured for **quick reference during contests**.

---

# 📘 Geometry Formulas for Competitive Programming

---

## 🔹 1. Basic 2D Geometry

* **Distance between two points**

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

* **Midpoint**

$$
M = \left(\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2}\right)
$$

* **Slope of line**

$$
m = \frac{y_2 - y_1}{x_2 - x_1}, \quad x_1 \neq x_2
$$

* **Equation of line through (x1, y1) & (x2, y2)**

$$
(y - y_1)(x_2 - x_1) = (y_2 - y_1)(x - x_1)
$$

* **Point-line distance** (line $ax+by+c=0$):

$$
d = \frac{|ax_0 + by_0 + c|}{\sqrt{a^2+b^2}}
$$

* **Dot product**

$$
\vec{a}\cdot \vec{b} = |a||b|\cos\theta = x_1x_2 + y_1y_2
$$

* **Cross product (2D)**

$$
\vec{a}\times \vec{b} = x_1y_2 - x_2y_1
$$

Used for orientation tests (left turn/right turn).

---

## 🔹 2. Triangle Geometry

* **Area (Heron’s formula)**

$$
A = \sqrt{s(s-a)(s-b)(s-c)}, \quad s = \frac{a+b+c}{2}
$$

* **Area using coordinates**

$$
A = \frac{1}{2}\, |x_1(y_2-y_3) + x_2(y_3-y_1) + x_3(y_1-y_2)|
$$

* **Area using cross product**

$$
A = \frac{1}{2} |\vec{AB} \times \vec{AC}|
$$

* **Law of Cosines**

$$
c^2 = a^2 + b^2 - 2ab\cos C
$$

* **Law of Sines**

$$
\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} = 2R
$$

* **Inradius**

$$
r = \frac{A}{s}
$$

* **Circumradius**

$$
R = \frac{abc}{4A}
$$

* **Centroid**

$$
G = \left(\frac{x_1+x_2+x_3}{3}, \frac{y_1+y_2+y_3}{3}\right)
$$

* **Incenter (weighted by sides)**

$$
I = \left(\frac{ax_1+bx_2+cx_3}{a+b+c}, \frac{ay_1+by_2+cy_3}{a+b+c}\right)
$$

* **Orthocenter** → intersection of altitudes.
* **Circumcenter** → intersection of perpendicular bisectors.

---

## 🔹 3. Polygon Geometry

* **Shoelace formula (area of polygon)**

$$
A = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - x_{i+1} y_i) \right|
$$

(where $x_{n+1}=x_1, y_{n+1}=y_1$).

* **Convex polygon perimeter**

$$
P = \sum_{i=1}^{n} \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}
$$

* **Pick’s theorem (lattice polygon)**

$$
A = I + \frac{B}{2} - 1
$$

(I = interior lattice points, B = boundary lattice points)

---

## 🔹 4. Circle Geometry

* **Equation of circle** (center $(h,k)$, radius $r$):

$$
(x-h)^2 + (y-k)^2 = r^2
$$

* **Chord length**

$$
L = 2\sqrt{r^2 - d^2}, \quad d=\text{distance from center to chord}
$$

* **Sector area**

$$
A = \frac{\theta}{2\pi} \cdot \pi r^2 = \frac{1}{2} r^2 \theta
$$

* **Arc length**

$$
s = r \theta
$$

* **Power of a point**

$$
PA \cdot PB = PC \cdot PD
$$

(for chords through a point $P$)

---

## 🔹 5. 3D Geometry

* **Distance between two points**

$$
d = \sqrt{(x_2-x_1)^2+(y_2-y_1)^2+(z_2-z_1)^2}
$$

* **Dot product**

$$
\vec{a}\cdot \vec{b} = x_1x_2 + y_1y_2 + z_1z_2
$$

* **Cross product**

$$
\vec{a}\times \vec{b} =
\begin{vmatrix}
i & j & k \\
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2
\end{vmatrix}
$$

* **Volume of parallelepiped**

$$
V = |\vec{a}\cdot(\vec{b}\times \vec{c})|
$$

* **Distance from point to plane (ax+by+cz+d=0)**

$$
d = \frac{|ax_0+by_0+cz_0+d|}{\sqrt{a^2+b^2+c^2}}
$$

---

## 🔹 6. Special Formulas & Algorithms

* **Convex Hull**:

  * Graham scan / Andrew monotone chain: $O(n \log n)$.
  * Useful for enclosing polygons.

* **Rotating Calipers**:

  * Maximum distance between points in convex polygon.
  * Minimum bounding rectangle.

* **Circle through 3 points** (circumcircle):
  Solve intersection of perpendicular bisectors.

---

## 🔹 7. Trigonometric Identities (CP-useful)

* $\sin^2\theta + \cos^2\theta = 1$
* $\sin(2\theta) = 2\sin\theta\cos\theta$
* $\cos(2\theta) = \cos^2\theta - \sin^2\theta$
* $\tan(A\pm B) = \frac{\tan A \pm \tan B}{1 \mp \tan A \tan B}$

---

✅ This covers all **essential geometry formulas for CP**:

* 2D basics (distances, dot/cross product, line eqn)
* Triangle formulas (area, inradius, circumradius, laws)
* Polygon formulas (shoelace, Pick’s theorem)
* Circle geometry (sector, chord, power of point)
* 3D geometry (distance, volume, plane formulas)
* Convex hull + rotating calipers (contest algorithms)

---

👉 Do you want me to now **add Python geometry templates** (distance, area, convex hull, polygon area, circle intersection) alongside formulas, so you can directly use them in contests?
