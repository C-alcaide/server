#include "vmx.h"
#include "vmx_producer.h"
#include "vmx_consumer.h"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <common/env.h>
#include <common/utf.h>

namespace caspar { namespace vmx {

namespace {

spl::shared_ptr<core::frame_producer> create_producer(const core::frame_producer_dependencies& dependencies, const std::vector<std::wstring>& params)
{
    if (params.size() == 0) return core::frame_producer::empty();
    
    auto path_w = params[0];

    // Auto-append logic:
    // If it doesn't end in .mav, we should check if file exists with .mav
    // OR just let the producer handle it if we want lazy checking?
    // But Core usually iterates producers to find one that accepts the file.
    // If we return empty, next producer tries.
    // So if user types "rec_basket", we must detect if "rec_basket.mav" exists or if it is a VMX intention.

    bool is_mav = boost::algorithm::iends_with(path_w, L".mav") || boost::algorithm::iends_with(path_w, L".MAV");
    
    if (!is_mav) {
         // Check if file.mav exists
         boost::filesystem::path p(path_w);
         if (p.is_absolute()) {
             if (boost::filesystem::exists(p.string() + ".mav")) {
                 path_w += L".mav";
                 is_mav = true;
             }
         } else {
             // Relative to media
             boost::filesystem::path m(env::media_folder());
             m /= p;
             if (boost::filesystem::exists(m.string() + ".mav")) {
                 path_w += L".mav";
                 is_mav = true;
             }
         }
    }

    if (!is_mav) return core::frame_producer::empty();
    
    auto path = u8(path_w);
    auto p = spl::make_shared<vmx_producer>(path, dependencies.frame_factory);
    
    // Configure producer based on params (skip filename at 0)
    if (params.size() > 1) {
        // Collect params into a vector or just pass params?
        // create_producer receives all params.
        p->configure(params);
    }
    
    return p;
}

spl::shared_ptr<core::frame_consumer> create_consumer(const std::vector<std::wstring>& params,
                                                  const core::video_format_repository& format_repository,
                                                  const std::vector<spl::shared_ptr<core::video_channel>>& channels,
                                                  const core::channel_info& channel_info)
{
    // Usage: ADD 1 VMX file [QUALITY]
    // params[0] = VMX
    // params[1] = filename
    if (params.size() < 2 || (!boost::iequals(params.at(0), L"VMX"))) 
         return core::frame_consumer::empty();
    
    // Careful: When removing, REMOVE 1 VMX file is called.
    // The consumer factory is used to MATCH the consumer to remove as well?
    // In CasparCG, create_consumer is called to Create. 
    // To Remove, Core matches existing consumers.
    
    // We must handle the filename similarly to producer regarding extension
    
    auto path_in = params[1];
    
    // If user passed filename, ensure .mav is appended if missing?
    // Or do we strictly require .mav?
    // Let's be lenient like producer
    
    // Check if query string "file?query" exists, append .mav before ?
    std::wstring path_w = path_in;
    std::wstring query_part = L"";
    size_t q_pos = path_w.find(L'?');
    if (q_pos != std::wstring::npos) {
        query_part = path_w.substr(q_pos);
        path_w = path_w.substr(0, q_pos);
    }

    if (!boost::algorithm::iends_with(path_w, L".mav") && !boost::algorithm::iends_with(path_w, L".MAV")) {
        path_w += L".mav";
    }
    
    auto path = u8(path_w + query_part);
    
    VMX_PROFILE quality = VMX_PROFILE_SQ;
    
    if (params.size() > 2)
    {
        std::wstring q = params[2];
        if (boost::iequals(q, L"LQ")) quality = VMX_PROFILE_LQ;
        else if (boost::iequals(q, L"SQ")) quality = VMX_PROFILE_SQ;
        else if (boost::iequals(q, L"HQ")) quality = VMX_PROFILE_HQ;
        else if (boost::iequals(q, L"OMT_LQ")) quality = VMX_PROFILE_OMT_LQ;
        else if (boost::iequals(q, L"OMT_SQ")) quality = VMX_PROFILE_OMT_SQ;
        else if (boost::iequals(q, L"OMT_HQ")) quality = VMX_PROFILE_OMT_HQ;
    }
    
    return spl::make_shared<vmx_consumer>(path, quality);
}

}

void init(const core::module_dependencies& dependencies)
{
    dependencies.producer_registry->register_producer_factory(L"VMX Producer", create_producer);
    dependencies.consumer_registry->register_consumer_factory(L"VMX Consumer", create_consumer);
}

void uninit()
{
}

}}
