// 360° Panorama Viewer using Three.js

class PanoViewer {
    constructor(container) {
        this.container = container;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.sphere = null;
        this.isUserInteracting = false;
        this.lon = 0;
        this.lat = 0;
        this.onPointerDownLon = 0;
        this.onPointerDownLat = 0;
        this.onPointerDownX = 0;
        this.onPointerDownY = 0;

        this.init();
    }

    init() {
        // Remove placeholder
        const placeholder = this.container.querySelector('.viewer-placeholder');

        // Scene
        this.scene = new THREE.Scene();

        // Camera
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1100);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        // Create inverted sphere for 360 viewing
        const geometry = new THREE.SphereGeometry(500, 60, 40);
        geometry.scale(-1, 1, 1);  // Invert for inside-out view

        // Default material (dark placeholder)
        const material = new THREE.MeshBasicMaterial({
            color: 0x333344
        });

        this.sphere = new THREE.Mesh(geometry, material);
        this.scene.add(this.sphere);

        // Event listeners
        this.container.addEventListener('pointerdown', (e) => this.onPointerDown(e));
        this.container.addEventListener('pointermove', (e) => this.onPointerMove(e));
        this.container.addEventListener('pointerup', () => this.onPointerUp());
        this.container.addEventListener('wheel', (e) => this.onWheel(e));

        window.addEventListener('resize', () => this.onResize());

        // Start render loop
        this.animate();
    }

    loadImage(url) {
        console.log('[PanoViewer] Loading image:', url);
        const loader = new THREE.TextureLoader();

        loader.load(
            url,
            (texture) => {
                console.log('[PanoViewer] Texture loaded successfully');
                texture.colorSpace = THREE.SRGBColorSpace;

                const material = new THREE.MeshBasicMaterial({
                    map: texture
                });

                this.sphere.material.dispose();
                this.sphere.material = material;

                // Remove placeholder if still present
                const placeholder = this.container.querySelector('.viewer-placeholder');
                if (placeholder) {
                    placeholder.style.display = 'none';
                }
            },
            (progress) => {
                console.log('[PanoViewer] Loading progress:', progress);
            },
            (error) => {
                console.error('[PanoViewer] Error loading texture:', error);
            }
        );
    }

    onPointerDown(event) {
        this.isUserInteracting = true;
        this.onPointerDownX = event.clientX;
        this.onPointerDownY = event.clientY;
        this.onPointerDownLon = this.lon;
        this.onPointerDownLat = this.lat;
    }

    onPointerMove(event) {
        if (!this.isUserInteracting) return;

        this.lon = (event.clientX - this.onPointerDownX) * 0.1 + this.onPointerDownLon;
        this.lat = (event.clientY - this.onPointerDownY) * 0.1 + this.onPointerDownLat;
        this.lat = Math.max(-85, Math.min(85, this.lat));
    }

    onPointerUp() {
        this.isUserInteracting = false;
    }

    onWheel(event) {
        const fov = this.camera.fov + event.deltaY * 0.05;
        this.camera.fov = Math.max(30, Math.min(100, fov));
        this.camera.updateProjectionMatrix();
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.update();
    }

    update() {
        // Convert spherical to Cartesian
        const phi = THREE.MathUtils.degToRad(90 - this.lat);
        const theta = THREE.MathUtils.degToRad(this.lon);

        const target = new THREE.Vector3(
            Math.sin(phi) * Math.cos(theta),
            Math.cos(phi),
            Math.sin(phi) * Math.sin(theta)
        );

        this.camera.lookAt(target);
        this.renderer.render(this.scene, this.camera);
    }

    dispose() {
        if (this.renderer) {
            this.renderer.dispose();
        }
        if (this.sphere) {
            this.sphere.geometry.dispose();
            this.sphere.material.dispose();
        }
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PanoViewer;
}
