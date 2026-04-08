use crate::Extractor;
use crate::data_reader::DataReader;
use crate::ncnn_bind::*;
use std::ffi::CString;

pub struct Net {
    ptr: ncnn_net_t,
}

unsafe impl Send for Net {}
unsafe impl Sync for Net {}

impl Net {
    pub fn new() -> Net {
        Net {
            ptr: unsafe { ncnn_net_create() },
        }
    }

    pub fn set_option(&mut self, opt: &crate::option::Option) {
        unsafe {
            ncnn_net_set_option(self.ptr, opt.ptr());
        }
    }

    pub fn set_vulkan_device(&mut self, _device_index: u32) {
        #[cfg(not(feature = "cpu"))]
        {
            use std::ffi::c_int;
            unsafe {
                ncnn_net_set_vulkan_device(self.ptr, _device_index as c_int);
            }
        }
    }

    pub fn load_param(&mut self, path: &str) -> anyhow::Result<()> {
        let c_str = {
            #[cfg(target_os = "windows")]
            {
                let (gbk_bytes, _, _) = encoding_rs::GB18030.encode(path);
                CString::new(gbk_bytes)?
            }
            #[cfg(not(target_os = "windows"))]
            {
                CString::new(path)?
            }
        };
        if unsafe { ncnn_net_load_param(self.ptr, c_str.as_ptr()) } == 0 {
            return Ok(());
        }

        #[cfg(target_os = "windows")] // 当 Windows 为 utf-8 编码时，再尝试一次
        {
            let c_str = CString::new(path)?;
            if unsafe { ncnn_net_load_param(self.ptr, c_str.as_ptr()) } == 0 {
                return Ok(());
            }
        }

        anyhow::bail!("Error loading params {}", path);
    }

    pub fn load_param_memory(&mut self, param_data: &[u8]) -> anyhow::Result<()> {
        let c_str =
            CString::new(param_data).map_err(|e| anyhow::anyhow!("Invalid param data: {}", e))?;
        let result = unsafe { ncnn_net_load_param_memory(self.ptr, c_str.as_ptr()) };
        if result != 0 {
            anyhow::bail!("Error loading params from memory");
        } else {
            Ok(())
        }
    }
    pub fn load_model(&mut self, path: &str) -> anyhow::Result<()> {
        let c_str = {
            #[cfg(target_os = "windows")]
            {
                let (gbk_bytes, _, _) = encoding_rs::GB18030.encode(path);
                CString::new(gbk_bytes)?
            }
            #[cfg(not(target_os = "windows"))]
            {
                CString::new(path)?
            }
        };
        if unsafe { ncnn_net_load_model(self.ptr, c_str.as_ptr()) } == 0 {
            return Ok(());
        }

        #[cfg(target_os = "windows")] // 当 Windows 为 utf-8 编码时，再尝试一次
        {
            let c_str = CString::new(path)?;
            if unsafe { ncnn_net_load_model(self.ptr, c_str.as_ptr()) } == 0 {
                return Ok(());
            }
        }

        anyhow::bail!("Error loading model {}", path);
    }

    pub fn load_model_datareader(&mut self, dr: &DataReader) -> anyhow::Result<()> {
        if unsafe { ncnn_net_load_model_datareader(self.ptr, dr.ptr()) } != 0 {
            anyhow::bail!("Error loading model from datareader");
        } else {
            Ok(())
        }
    }

    pub fn create_extractor(&self) -> Extractor<'_> {
        let ptr;
        unsafe {
            ptr = ncnn_extractor_create(self.ptr);
        }
        Extractor::from_ptr(ptr)
    }
}

impl Drop for Net {
    fn drop(&mut self) {
        unsafe {
            ncnn_net_destroy(self.ptr);
        }
    }
}
