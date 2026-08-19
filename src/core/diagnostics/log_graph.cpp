/*
 * Copyright (c) 2011 Sveriges Television AB <info@casparcg.com>
 *
 * This file is part of CasparCG (www.casparcg.com).
 *
 * CasparCG is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CasparCG is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CasparCG. If not, see <http://www.gnu.org/licenses/>.
 */

#include "../StdAfx.h"

#include "log_graph.h"

#include <common/diagnostics/graph.h>
#include <common/env.h>
#include <common/log.h>
#include <common/memory.h>
#include <common/timer.h>
#include <common/utf.h>

#include <boost/property_tree/ptree.hpp>

#include <algorithm>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace caspar { namespace core { namespace diagnostics { namespace log {

namespace {

/// One sink per graph -- the SPI calls the factory once per registered graph, exactly as the
/// OSD window does, so per-graph state needs no keying.
class log_sink : public caspar::diagnostics::spi::graph_sink
{
    mutable std::mutex            mutex_;
    std::wstring                  text_;
    std::map<std::string, double> values_;
    // Tags are EVENTS, not levels: a dropped frame happens once and is gone. Counting them
    // between emits is the whole point -- a queue that never reads empty while frames are
    // being dropped looks healthy on the values alone.
    std::map<std::string, int>    tags_;
    caspar::timer                 since_emit_;
    double                        interval_;
    bool                          have_data_ = false;

  public:
    explicit log_sink(double interval)
        : interval_(interval)
    {
    }

    void activate() override {}

    void set_text(const std::wstring& value) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        text_ = value;
    }

    void set_value(const std::string& name, double value) override
    {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[name] = value;
        have_data_    = true;
        emit_if_due();
    }

    // Colour is presentation only; it says nothing a measurement can use.
    void set_color(const std::string& /*name*/, int /*color*/) override {}

    void set_tag(caspar::diagnostics::tag_severity severity, const std::string& name) override
    {
        if (severity == caspar::diagnostics::tag_severity::SILENT)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        ++tags_[name];
        have_data_ = true;
    }

    void auto_reset() override {}

  private:
    /// Emitted from set_value rather than a timer thread: every graph that matters is being
    /// written to on the frame path anyway, so this costs no thread and cannot outlive the
    /// component it describes. A graph that stops being written stops logging, which is
    /// itself the signal you want.
    void emit_if_due()
    {
        if (since_emit_.elapsed() < interval_ || !have_data_)
            return;

        std::wostringstream line;
        line << L"[diag] ";
        if (!text_.empty())
            line << text_ << L"  ";

        // Sorted, so successive lines for one graph are column-comparable by eye and by script.
        bool first = true;
        for (const auto& [name, value] : values_) {
            if (!first)
                line << L' ';
            first = false;
            line << u16(name) << L'=' << std::fixed << std::setprecision(4) << value;
        }

        if (!tags_.empty()) {
            line << L"  | tags:";
            for (const auto& [name, count] : tags_) {
                line << L' ' << u16(name);
                if (count > 1)
                    line << L"x" << count;
            }
        }

        CASPAR_LOG(info) << line.str();

        tags_.clear();          // events: reported once, then gone
        have_data_ = false;     // values persist; a silent graph must not repeat stale numbers
        since_emit_.restart();
    }
};

} // namespace

void register_sink()
{
    // Seconds between lines per graph. 1 s matches the OSD's readable cadence and is coarse
    // enough not to matter on the frame path; a boundary event (loop wrap, turnaround) is
    // better caught by the tags, which are counted rather than sampled.
    const double interval = std::max(0.05, env::properties().get(L"configuration.log-diagnostics-interval", 1.0));

    caspar::diagnostics::spi::register_sink_factory(
        [interval] { return spl::make_shared<log_sink>(interval); });

    CASPAR_LOG(info) << L"[diag] diagnostics logging enabled, one line per graph every "
                     << interval << L" s";
}

}}}} // namespace caspar::core::diagnostics::log
