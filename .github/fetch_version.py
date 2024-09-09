from setuptools_scm import get_version

def fetch_version():
    try:
        # You can add additional parameters here to match your pyproject.toml configuration
        version = get_version(
            root=".",  # Assuming you're running this from the project root
            version_scheme="only-version",
            local_scheme="no-local-version",
        )
        return version
    except Exception as e:
        print(f"Error fetching version: {e}")
        return None

if __name__ == "__main__":
    print(fetch_version())